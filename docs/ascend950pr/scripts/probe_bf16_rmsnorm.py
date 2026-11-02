"""MINIMAL bf16 RMSNorm feasibility probe on Ascend950PR.

Purpose: answer one question the plan depends on - can TileLang-Ascend compile and run a
BF16-in/BF16-out RMSNorm with FP32 statistics, matching the frozen task contract?
Derived from the repository's examples/ascend/example_rmsnorm.py (fp32), adapted to bf16
I/O. Read-only with respect to all three repositories: only writes to a private cache.
"""
import argparse, json, statistics, sys
import torch, torch_npu, tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench

def rms_norm_fwd(batch, d, in_dtype="bfloat16"):
    N_CORES = 64
    TILE = tilelang.next_power_of_2(d)
    ub_budget = 250 * 1024  # measured UB ceiling is 256 KiB; keep margin for compiler static shared memory
    rows = 4
    while rows > 1 and (batch % (N_CORES * rows) or (rows * d * 2 + rows * d * 4 * 2) > ub_budget):
        rows //= 2
    threads = 256 if rows == 4 else (512 if rows == 2 else 128)
    threads = min(threads, TILE)
    iterations = batch // (N_CORES * rows)
    N = batch * d

    @T.prim_func
    def main(X: T.Tensor((N,), in_dtype), Y: T.Tensor((N,), in_dtype), W: T.Tensor((d,), in_dtype), eps: T.float32):
        with T.Kernel(N_CORES) as core_id:
            w_ub = T.alloc_shared((d,), in_dtype)
            x_ub = T.alloc_shared(rows * d, in_dtype)
            y_ub = T.alloc_shared(rows * d, in_dtype)
            T.annotate_buffer_versions({x_ub: 2, y_ub: 1})
            T.copy(W[:d], w_ub[:d])
            for t in T.Pipelined(iterations, num_stages=2):
                row_id = (t * N_CORES + core_id) * rows
                base = row_id * d
                T.copy(X[base : base + rows * d], x_ub)
                with T.SimtVF(threads=threads):
                    x_frag = T.alloc_fragment((rows, TILE), "float32")
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        x_frag[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))
                    w_frag = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE, coalesced_width=T.int32(1)):
                        w_frag[i] = T.if_then_else(i < d, w_ub[i], T.float32(0))
                    sum_sq = T.alloc_reducer((rows,), "float32", op="sum")
                    T.reducer_init(sum_sq)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            T.reducer_update(sum_sq[r], x_frag[r, i] * x_frag[r, i])
                    sum_result = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(sum_sq, sum_result)
                    for r in T.Parallel(rows):
                        sum_result[r] = T.rsqrt(sum_result[r] / d + eps)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            y_ub[r * d + i] = T.cast(x_frag[r, i] * sum_result[r] * w_frag[i], in_dtype)
                T.copy(y_ub, Y[base : base + rows * d])
    return main

def ref(x, w, eps=1e-6):
    rstd = torch.rsqrt(x.float().pow(2).mean(-1) + eps)
    return (x.float() * rstd.unsqueeze(-1) * w.float().unsqueeze(0)).to(torch.bfloat16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    out = {"batch": a.batch, "d": a.d, "dtype": "bfloat16"}
    x = torch.randn(a.batch, a.d, dtype=torch.bfloat16, device="npu")
    w = torch.randn(a.d, dtype=torch.bfloat16, device="npu")
    kernel = tilelang.compile(rms_norm_fwd(a.batch, a.d), target="ascend", out_idx=[1])
    y = kernel(x.flatten(), w, 1e-6).reshape(a.batch, a.d)
    r = ref(x, w)
    diff = (y.float() - r.float()).abs()
    denom = r.float().abs().clamp_min(1e-6)
    out["compiled"] = True
    out["max_abs_err"] = float(diff.max())
    out["max_rel_err"] = float((diff / denom).max())
    v = torch_npu.npu_rms_norm(x, w, 1e-6)[0]
    vd = (v.float() - r.float()).abs()
    out["vendor_max_abs_err"] = float(vd.max())
    out["vendor_max_rel_err"] = float((vd / denom).max())
    if a.bench:
        def med(fn):
            fn()
            value = do_bench(fn, backend="msprof", _n_warmup=10, _n_repeat=30)
            return statistics.median(value) if isinstance(value, (list, tuple)) else float(value)
        out["tilelang_ms"] = med(lambda: kernel(x.flatten(), w, 1e-6))
        out["vendor_ms"] = med(lambda: torch_npu.npu_rms_norm(x, w, 1e-6)[0])
        out["speedup_vs_vendor"] = out["vendor_ms"] / out["tilelang_ms"]
    print("PROBE_JSON " + json.dumps(out))

main()
