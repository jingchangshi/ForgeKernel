"""Show that the naive tolerance gate cannot separate a correct kernel from a sloppy one.

The calibrate MATCH floor is 2^-7 = 0.0078125, which equals bf16 unit-roundoff. A correct
FP32-accumulating kernel lands at ~0.0078 and so does a BF16-accumulating one, so a
tolerance-only gate has no discriminating power. This script measures both kernels against
a high-precision FP64 reference and reports the two error statistics that DO separate them:
  - max_abs / max_rel against FP64 (both saturate at bf16 rounding)
  - max_rel on a HELD-OUT near-cancellation family (where the two disagree)
"""
import json, torch, torch_npu, tilelang
import tilelang.ascend.language as T
from tilelang.profiler.msprof import bench_with_msprof

B, D = 256, 4096


def build(batch, d, accum):
    N_CORES, rows = 64, 4
    TILE = tilelang.next_power_of_2(d)
    iterations = batch // (N_CORES * rows)
    N = batch * d
    acc_dtype = "float32" if accum == "fp32" else "bfloat16"

    @T.prim_func
    def main(X: T.Tensor((N,), "bfloat16"), Y: T.Tensor((N,), "bfloat16"),
             W: T.Tensor((d,), "bfloat16"), eps: T.float32):
        with T.Kernel(N_CORES) as core_id:
            w_ub = T.alloc_shared((d,), "bfloat16")
            x_ub = T.alloc_shared(rows * d, "bfloat16")
            y_ub = T.alloc_shared(rows * d, "bfloat16")
            T.annotate_buffer_versions({x_ub: 2, y_ub: 1})
            T.copy(W[:d], w_ub[:d])
            for t in T.Pipelined(iterations, num_stages=2):
                row_id = (t * N_CORES + core_id) * rows
                base = row_id * d
                T.copy(X[base:base + rows * d], x_ub)
                with T.SimtVF(threads=256):
                    x_frag = T.alloc_fragment((rows, TILE), acc_dtype)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        x_frag[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.cast(0, acc_dtype))
                    w_frag = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE, coalesced_width=T.int32(1)):
                        w_frag[i] = T.if_then_else(i < d, w_ub[i], T.float32(0))
                    ss = T.alloc_reducer((rows,), acc_dtype, op="sum")
                    T.reducer_init(ss)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            T.reducer_update(ss[r], x_frag[r, i] * x_frag[r, i])
                    # finalize_reducer requires the destination to match the reducer
                    # dtype, so a bf16-accumulating kernel also carries its rstd in
                    # bf16 - which is exactly the sloppiness under test.
                    res = T.alloc_fragment((rows,), acc_dtype)
                    T.finalize_reducer(ss, res)
                    for r in T.Parallel(rows):
                        res[r] = T.cast(T.rsqrt(T.cast(res[r], "float32") / d + eps), acc_dtype)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            y_ub[r * d + i] = T.cast(T.cast(x_frag[r, i], "float32") * res[r] * w_frag[i], "bfloat16")
                T.copy(y_ub, Y[base:base + rows * d])
    return main


def reference(x, w, eps=1e-6):
    """FP64 reference: the definition, computed out of reach of bf16 rounding."""
    xd = x.double()
    rstd = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
    return (xd * rstd * w.double().unsqueeze(0)).to(torch.bfloat16)


def stats(out, exp, tag):
    o, e = out.float(), exp.float()
    diff = (o - e).abs()
    denom = e.abs().clamp_min(1e-6)
    return {tag + "_max_abs": round(float(diff.max()), 7),
            tag + "_max_rel": round(float((diff / denom).max()), 7),
            tag + "_mean_abs": round(float(diff.mean()), 9)}


w = torch.randn(D, dtype=torch.bfloat16, device="npu")
# Held-out family: near-cancellation rows where a low-precision accumulator must drift.
torch.manual_seed(7)
x_norm = torch.randn(B, D, dtype=torch.bfloat16, device="npu")
x_tiny = (torch.randn(B, D, device="npu") * 1e-3).to(torch.bfloat16)
x_out = torch.randn(B, D, device="npu").to(torch.bfloat16)
x_out[:, 0] = 300.0
families = {"normal": x_norm, "near_zero_rows": x_tiny, "outlier_channels": x_out}

out = {"batch": B, "d": D, "tolerance_floor": 2 ** -7, "families": {}}
for name, x in families.items():
    exp = reference(x, w)
    fam = {}
    for accum in ("fp32", "bf16"):
        kern = tilelang.compile(build(B, D, accum), target="ascend", out_idx=[1])
        y = kern(x.flatten(), w, 1e-6).reshape(B, D)
        fam.update(stats(y, exp, accum))
    fam["fp32_passes_naive_tolerance"] = fam["fp32_max_rel"] <= out["tolerance_floor"]
    fam["bf16_passes_naive_tolerance"] = fam["bf16_max_rel"] <= out["tolerance_floor"]
    fam["separation_factor"] = (round(fam["bf16_max_rel"] / fam["fp32_max_rel"], 2)
                                if fam["fp32_max_rel"] > 0 else None)
    out["families"][name] = fam

print("PRECISION_JSON " + json.dumps(out))