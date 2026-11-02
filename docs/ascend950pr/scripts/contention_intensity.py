"""Reconcile two contradictory contention results.

Run 1 (xprocess_perturbation): a 2048x2048 matmul neighbour moved the ratio 0.251 -> 0.427
  (70% toward parity).
Run 2 (contention_bias): a 1024x1024 matmul neighbour moved it 0.2663 -> 0.2669 (0.2%).

Both cannot describe the same phenomenon. The difference is neighbour intensity and whether
the neighbour actually keeps the device busy. This script measures, for several neighbour
intensities, (a) how much each arm slows and (b) how much the ratio moves - so the protocol
can state when a window is trustworthy instead of assuming pairing always cancels.
"""
import json, statistics, subprocess, sys, textwrap, time

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


NEIGHBOUR = textwrap.dedent("""
    import sys, time, torch, torch_npu
    seconds, n, inner = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    a = torch.randn(n, n, dtype=torch.bfloat16, device="npu")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="npu")
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(inner):
            torch.matmul(a, b)
        torch.npu.synchronize()
""")

x = torch.randn(B, D, dtype=torch.bfloat16, device="npu")
w = torch.randn(D, dtype=torch.bfloat16, device="npu")
kern = tilelang.compile(build(B, D), target="ascend", out_idx=[1])
cache = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device="npu")


def tiled():
    kern(x.flatten(), w, 1e-6)


def vendor():
    torch_npu.npu_rms_norm(x, w, 1e-6)


tiled(); vendor(); torch.npu.synchronize()


def window(n=7):
    t, v, p = [], [], []
    for k in range(n):
        vals = {}
        for fn in ((tiled, vendor) if k % 2 == 0 else (vendor, tiled)):
            vals["t" if fn is tiled else "v"] = bench_with_msprof(fn, cache, 10)
        t.append(vals["t"]); v.append(vals["v"]); p.append(vals["v"] / vals["t"])
    return statistics.median(t), statistics.median(v), statistics.median(p)


base_t, base_v, base_ratio = window()
arms = []
for label, size, inner in (("light_1024_x40", 1024, 40), ("heavy_2048_x20", 2048, 20)):
    child = subprocess.Popen([sys.executable, "-c", NEIGHBOUR, "50", str(size), str(inner)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6.0)
    t, v, r = window()
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
    time.sleep(2.0)
    arms.append({"neighbour": label,
                 "tiled_ms": round(t, 6), "vendor_ms": round(v, 6), "ratio": round(r, 4),
                 "tiled_slowdown_pct": round(100 * (t - base_t) / base_t, 1),
                 "vendor_slowdown_pct": round(100 * (v - base_v) / base_v, 1),
                 "ratio_shift_pct": round(100 * (r - base_ratio) / base_ratio, 1)})

print("INTENSITY_JSON " + json.dumps({
    "shape": "%dx%d bf16" % (B, D),
    "baseline": {"tiled_ms": round(base_t, 6), "vendor_ms": round(base_v, 6),
                 "ratio": round(base_ratio, 4)},
    "arms": arms,
}))