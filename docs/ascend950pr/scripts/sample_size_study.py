"""How many pairs does a verdict need? Answer it by resampling measured data.

The acceptance protocol judges on a paired bootstrap lower bound. Its sample size was a
guess. This measures real paired samples under normal co-tenant activity, then resamples
them at increasing n to show how the interval width and the stability of the verdict
depend on the number of pairs - so the protocol's defaults are chosen from evidence.
"""
import json, statistics, torch, torch_npu, tilelang
import tilelang.ascend.language as T
from tilelang.profiler.msprof import bench_with_msprof

B, D = 4096, 4096


def build(batch, d):
    N_CORES, rows = 64, 4
    TILE = tilelang.next_power_of_2(d)
    iters = batch // (N_CORES * rows)
    N = batch * d

    @T.prim_func
    def main(X: T.Tensor((N,), "bfloat16"), Y: T.Tensor((N,), "bfloat16"),
             W: T.Tensor((d,), "bfloat16"), eps: T.float32):
        with T.Kernel(N_CORES) as core_id:
            w_ub = T.alloc_shared((d,), "bfloat16")
            x_ub = T.alloc_shared(rows * d, "bfloat16")
            y_ub = T.alloc_shared(rows * d, "bfloat16")
            T.annotate_buffer_versions({x_ub: 2, y_ub: 1})
            T.copy(W[:d], w_ub[:d])
            for t in T.Pipelined(iters, num_stages=2):
                base = ((t * N_CORES + core_id) * rows) * d
                T.copy(X[base:base + rows * d], x_ub)
                with T.SimtVF(threads=256):
                    xf = T.alloc_fragment((rows, TILE), "float32")
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        xf[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))
                    wf = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE, coalesced_width=T.int32(1)):
                        wf[i] = T.if_then_else(i < d, w_ub[i], T.float32(0))
                    ss = T.alloc_reducer((rows,), "float32", op="sum")
                    T.reducer_init(ss)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            T.reducer_update(ss[r], xf[r, i] * xf[r, i])
                    res = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(ss, res)
                    for r in T.Parallel(rows):
                        res[r] = T.rsqrt(res[r] / d + eps)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            y_ub[r * d + i] = T.cast(xf[r, i] * res[r] * wf[i], "bfloat16")
                T.copy(y_ub, Y[base:base + rows * d])
    return main


x = torch.randn(B, D, dtype=torch.bfloat16, device="npu")
w = torch.randn(D, dtype=torch.bfloat16, device="npu")
kern = tilelang.compile(build(B, D), target="ascend", out_idx=[1])
cache = torch.empty(32 * 1024 * 1024 // 4, dtype=torch.int32, device="npu")


def tiled():
    kern(x.flatten(), w, 1e-6)


def vendor():
    torch_npu.npu_rms_norm(x, w, 1e-6)


tiled(); vendor(); torch.npu.synchronize()

pairs = []
N_PAIRS = 25
for k in range(N_PAIRS):
    vals = {}
    for fn in ((tiled, vendor) if k % 2 == 0 else (vendor, tiled)):
        vals["t" if fn is tiled else "v"] = bench_with_msprof(fn, cache, 10)
    pairs.append(vals["v"] / vals["t"])


def bootstrap(pool, n, resamples=1500, seed=7):
    import random
    rng = random.Random(seed)
    meds = []
    for _ in range(resamples):
        sample = [pool[rng.randrange(len(pool))] for _ in range(n)]
        meds.append(statistics.median(sample))
    meds.sort()
    return meds[int(0.025 * resamples)], meds[int(0.975 * resamples)]


shape = []
for n in (5, 7, 9, 11, 15, 21):
    lo, hi = bootstrap(pairs, n)
    point = statistics.median(pairs[:n])
    shape.append({"pairs": n, "point": round(point, 4),
                  "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                  "ci_width": round(hi - lo, 4),
                  "width_pct_of_point": round(100 * (hi - lo) / point, 1)})

print("SAMPLESIZE_JSON " + json.dumps({
    "shape": "%dx%d bf16" % (B, D),
    "collected_pairs": N_PAIRS,
    "raw_pairs": [round(p, 4) for p in pairs],
    "median_ratio": round(statistics.median(pairs), 4),
    "min": round(min(pairs), 4), "max": round(max(pairs), 4),
    "by_sample_size": shape,
}))