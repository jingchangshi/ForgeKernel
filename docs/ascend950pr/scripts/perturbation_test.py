"""Quantify paired vs unpaired error when a neighbour loads the same card.

Case for the acceptance protocol on a shared machine: run alternating windows with a
sustained neighbour matmul stream on the same card, and report what two independent
medians would conclude versus what the paired median concludes.
"""
import json, statistics, threading, time

import torch, torch_npu, tilelang
import tilelang.ascend.language as T
from tilelang.profiler.msprof import bench_with_msprof

B, D = 4096, 4096


def build(batch, d):
    N_CORES, rows = 64, 4
    TILE = tilelang.next_power_of_2(d)
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
                with T.SimtVF(threads=256):
                    x_frag = T.alloc_fragment((rows, TILE), "float32")
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        x_frag[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))
                    w_frag = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE, coalesced_width=T.int32(1)):
                        w_frag[i] = T.if_then_else(i < d, w_ub[i], T.float32(0))
                    ss = T.alloc_reducer((rows,), "float32", op="sum")
                    T.reducer_init(ss)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            T.reducer_update(ss[r], x_frag[r, i] * x_frag[r, i])
                    res = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(ss, res)
                    for r in T.Parallel(rows):
                        res[r] = T.rsqrt(res[r] / d + eps)
                    for r, i in T.Parallel(rows, TILE, coalesced_width=T.int32(1)):
                        if i < d:
                            y_ub[r * d + i] = T.cast(x_frag[r, i] * res[r] * w_frag[i], "bfloat16")
                T.copy(y_ub, Y[base:base + rows * d])
    return main


x = torch.randn(B, D, dtype=torch.bfloat16, device="npu")
w = torch.randn(D, dtype=torch.bfloat16, device="npu")
kern = tilelang.compile(build(B, D), target="ascend", out_idx=[1])
cache = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device="npu")


def tiled():
    kern(x.flatten(), w, 1e-6)


def vendor():
    torch_npu.npu_rms_norm(x, w, 1e-6)


tiled(); vendor(); torch.npu.synchronize()

quiet_t, quiet_v = [], []
for k in range(5):
    order = (tiled, vendor) if k % 2 == 0 else (vendor, tiled)
    for fn in order:
        ms = bench_with_msprof(fn, cache, 10)
        (quiet_t if fn is tiled else quiet_v).append(ms)
quiet_ratio = statistics.median(quiet_v) / statistics.median(quiet_t)

stop = threading.Event()


def neighbour():
    a = torch.randn(2048, 2048, dtype=torch.bfloat16, device="npu")
    b = torch.randn(2048, 2048, dtype=torch.bfloat16, device="npu")
    while not stop.is_set():
        for _ in range(20):
            torch.matmul(a, b)
        torch.npu.synchronize()


thread = threading.Thread(target=neighbour, daemon=True)
thread.start()
time.sleep(2.0)

busy_t, busy_v, pairs = [], [], []
for k in range(11):
    order = (tiled, vendor) if k % 2 == 0 else (vendor, tiled)
    vals = {}
    for fn in order:
        vals["tiled" if fn is tiled else "vendor"] = bench_with_msprof(fn, cache, 10)
    busy_t.append(vals["tiled"])
    busy_v.append(vals["vendor"])
    pairs.append(vals["vendor"] / vals["tiled"])
stop.set(); thread.join(timeout=10)

unpaired = statistics.median(busy_v) / statistics.median(busy_t)
paired = statistics.median(pairs)
print("PERTURB_JSON " + json.dumps({
    "shape": "%dx%d bf16" % (B, D),
    "quiet_paired_ratio": round(quiet_ratio, 4),
    "busy_unpaired_ratio_of_medians": round(unpaired, 4),
    "busy_paired_ratio": round(paired, 4),
    "unpaired_error_vs_quiet_pct": round(100 * abs(unpaired - quiet_ratio) / quiet_ratio, 1),
    "paired_error_vs_quiet_pct": round(100 * abs(paired - quiet_ratio) / quiet_ratio, 1),
    "busy_tiled_median_ms": round(statistics.median(busy_t), 6),
    "busy_vendor_median_ms": round(statistics.median(busy_v), 6),
    "busy_pairs": [round(p, 4) for p in pairs],
}))