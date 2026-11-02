"""Test whether the example kernel is limited by per-core row tiling, not by the vector math.

Pipe counters say the candidate is starved on MTE2 (32%) while the vendor keeps it at 64%.
That is the signature of too few in-flight bulk copies per core. This sweeps the row tile
(`rows` rows per core per iteration) and reports device time per configuration, using the
same msprof basis the acceptance protocol uses.

Read-only: compiles from this script, writes only to a private cache directory.
"""
import argparse, json, statistics, torch, torch_npu, tilelang
import tilelang.ascend.language as T
from tilelang.profiler.msprof import bench_with_msprof


def build(batch, d, rows):
    N_CORES = 64
    TILE = tilelang.next_power_of_2(d)
    threads = min(256 if rows >= 4 else (512 if rows == 2 else 128), TILE)
    iterations = batch // (N_CORES * rows)
    N = batch * d

    @T.prim_func
    def main(X: T.Tensor((N,), "bfloat16"), Y: T.Tensor((N,), "bfloat16"),
             W: T.Tensor((d,), "bfloat16"), eps: T.float32):
        with T.Kernel(N_CORES) as core_id:
            w_ub = T.alloc_shared((d,), "bfloat16")
            x_ub = T.alloc_shared(rows * d, "bfloat16")
            y_ub = T.alloc_shared(rows * d, "bfloat16")
            T.annotate_buffer_versions({x_ub: 3, y_ub: 1})
            T.copy(W[:d], w_ub[:d])
            for t in T.Pipelined(iterations, num_stages=3):
                row_id = (t * N_CORES + core_id) * rows
                base = row_id * d
                T.copy(X[base:base + rows * d], x_ub)
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
                    res = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(sum_sq, res)
                    for r in T.Parallel(rows):
                        res[r] = T.rsqrt(res[r] / d + eps)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            y_ub[r * d + i] = T.cast(x_frag[r, i] * res[r] * w_frag[i], "bfloat16")
                T.copy(y_ub, Y[base:base + rows * d])
    return main


a = argparse.ArgumentParser()
a.add_argument("--batch", type=int, default=8192)
a.add_argument("--d", type=int, default=4096)
a.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8])
args = a.parse_args()

x = torch.randn(args.batch, args.d, dtype=torch.bfloat16, device="npu")
w = torch.randn(args.d, dtype=torch.bfloat16, device="npu")
cache = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device="npu")

def ref(x, w):
    rstd = torch.rsqrt(x.float().pow(2).mean(-1) + 1e-6)
    return (x.float() * rstd.unsqueeze(-1) * w.float().unsqueeze(0)).to(torch.bfloat16)

expected = ref(x, w)
results = []
for rows in args.rows:
    if args.batch % (64 * rows):
        results.append({"rows": rows, "skipped": "batch not divisible by 64*rows"})
        continue
    try:
        kern = tilelang.compile(build(args.batch, args.d, rows), target="ascend", out_idx=[1])
        y = kern(x.flatten(), w, 1e-6).reshape(args.batch, args.d)
        err = float((y.float() - expected.float()).abs().max())
        prof = bench_with_msprof(lambda: kern(x.flatten(), w, 1e-6), cache, 20, detailed=True)
        results.append({"rows": rows, "ms": round(prof.dur_ns / 1e6, 6),
                        "max_abs_err": err, "pipes": str(prof)[:120]})
    except Exception as exc:
        results.append({"rows": rows, "error": str(exc)[:200]})

vendor = bench_with_msprof(lambda: torch_npu.npu_rms_norm(x, w, 1e-6), cache, 20)
print("SWEEP_JSON " + json.dumps({"batch": args.batch, "d": args.d,
                                  "vendor_ms": round(vendor, 6), "results": results}))