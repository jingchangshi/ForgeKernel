"""Paired A/B benchmark on a SHARED card.

Validates the acceptance protocol under the user decision that other tenants may use
the same card. The claim to test: interleaving A/B and B/A inside one measurement window
and judging on the paired difference removes a drifting external load, whereas taking
two independent medians lets that load masquerade as a speed difference.

Two arms are measured:
  tiled   - the repository example TileLang bf16 RMSNorm (our candidate stand-in)
  vendor  - torch_npu.npu_rms_norm

Emits one JSON object with raw paired samples, per-arm medians, the paired median ratio,
and a bootstrap interval of that ratio. Read-only with respect to all repositories.
"""
import argparse, json, os, statistics

import torch, torch_npu, tilelang
import tilelang.ascend.language as T
from tilelang.profiler.msprof import bench_with_msprof


def rms_norm_bf16(batch, d):
    N_CORES = 64
    TILE = tilelang.next_power_of_2(d)
    rows = 4
    while rows > 1 and batch % (N_CORES * rows):
        rows //= 2
    threads = min(256 if rows == 4 else (512 if rows == 2 else 128), TILE)
    iterations = batch // (N_CORES * rows)
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
            for t in T.Pipelined(iterations, num_stages=2):
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


def bootstrap_ratio_ci(pairs, resamples=4000, seed=20261102):
    """95% interval of the median paired ratio vendor/tiled."""
    import random
    rng = random.Random(seed)
    n = len(pairs)
    stats = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        stats.append(statistics.median(sample))
    stats.sort()
    return round(stats[int(0.025 * resamples)], 4), round(stats[int(0.975 * resamples)], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--cache-mib", type=int, default=256)
    a = ap.parse_args()

    x = torch.randn(a.batch, a.d, dtype=torch.bfloat16, device="npu")
    w = torch.randn(a.d, dtype=torch.bfloat16, device="npu")
    kern = tilelang.compile(rms_norm_bf16(a.batch, a.d), target="ascend", out_idx=[1])
    # int32: msprof's cache-flush kernel cannot lower int8 vectors on Ascend.
    cache = torch.empty(a.cache_mib * 1024 * 1024 // 4, dtype=torch.int32, device="npu")

    def tiled():
        kern(x.flatten(), w, 1e-6)

    def vendor():
        torch_npu.npu_rms_norm(x, w, 1e-6)

    # Warm both arms before any sample so first-call compilation is not a sample.
    tiled(); vendor(); torch.npu.synchronize()

    tiled_ms, vendor_ms, pairs = [], [], []
    for k in range(a.pairs):
        order = (tiled, vendor) if k % 2 == 0 else (vendor, tiled)
        values = {}
        for fn in order:
            ms = bench_with_msprof(fn, cache, a.repeat)
            values["tiled" if fn is tiled else "vendor"] = ms
        tiled_ms.append(values["tiled"])
        vendor_ms.append(values["vendor"])
        pairs.append(values["vendor"] / values["tiled"])

    lo, hi = bootstrap_ratio_ci(pairs)
    out = {
        "shape": "%dx%d bf16" % (a.batch, a.d),
        "basis": "msprof per-kernel device time",
        "pairs": a.pairs,
        "repeat_per_pair": a.repeat,
        "shared_card": True,
        "npu_visible": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "unset"),
        "tiled_ms": [round(v, 6) for v in tiled_ms],
        "vendor_ms": [round(v, 6) for v in vendor_ms],
        "paired_ratio": [round(v, 4) for v in pairs],
        "tiled_median_ms": round(statistics.median(tiled_ms), 6),
        "vendor_median_ms": round(statistics.median(vendor_ms), 6),
        "unpaired_ratio_of_medians": round(statistics.median(vendor_ms) / statistics.median(tiled_ms), 4),
        "paired_median_ratio": round(statistics.median(pairs), 4),
        "paired_ratio_ci95": [lo, hi],
        "tiled_spread_pct": round(100 * (max(tiled_ms) - min(tiled_ms)) / statistics.median(tiled_ms), 1),
        "vendor_spread_pct": round(100 * (max(vendor_ms) - min(vendor_ms)) / statistics.median(vendor_ms), 1),
    }
    print("PAIRED_JSON " + json.dumps(out))


main()