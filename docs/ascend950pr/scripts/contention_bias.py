"""Confirm the direction of the contention bias, and whether it is detectable.

The prior result showed interleaved pairing does NOT cancel a co-tenant: the vendor arm
degraded 48% while the candidate arm degraded 3%, so the ratio moved 70% toward parity.
If that is systematic, measuring under contention makes a slower candidate look acceptable,
which is a false-PASS direction. This script checks reproducibility and detects the
contention level at the same time, so a window can be classified rather than trusted.
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
    seconds, inner = float(sys.argv[1]), int(sys.argv[2])
    a = torch.randn(1024, 1024, dtype=torch.bfloat16, device="npu")
    b = torch.randn(1024, 1024, dtype=torch.bfloat16, device="npu")
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
    return {"t": statistics.median(t), "v": statistics.median(v), "ratio": statistics.median(p)}


def timed(fn):
    torch.npu.synchronize(); start = time.perf_counter(); fn(); torch.npu.synchronize()
    return time.perf_counter() - start

quiet_a = window()
quiet_b = window()

child = subprocess.Popen([sys.executable, "-c", NEIGHBOUR, "60", "40"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(6.0)
busy = window()

# Cheap in-window contention probe: wall time of a fixed host round-trip rises under load.
probe = {"busy_wall_ms": round(1000 * statistics.median([timed(lambda: torch.npu.synchronize()) for _ in range(5)]), 3)}
child.terminate()
try:
    child.wait(timeout=5)
except subprocess.TimeoutExpired:
    child.kill()
time.sleep(2.0)
after = window()
probe["after_wall_ms"] = round(1000 * statistics.median([timed(lambda: torch.npu.synchronize()) for _ in range(5)]), 3)

print("BIAS_JSON " + json.dumps({
    "shape": "%dx%d bf16" % (B, D),
    "quiet_ratio_a": round(quiet_a["ratio"], 4),
    "quiet_ratio_b": round(quiet_b["ratio"], 4),
    "quiet_repeatability_pct": round(100 * abs(quiet_b["ratio"] - quiet_a["ratio"]) / quiet_a["ratio"], 2),
    "busy_ratio": round(busy["ratio"], 4),
    "busy_shift_toward_parity_pct": round(100 * (busy["ratio"] - quiet_a["ratio"]) / quiet_a["ratio"], 1),
    "after_ratio": round(after["ratio"], 4),
    "vendor_degradation_pct": round(100 * (busy["v"] - quiet_a["v"]) / quiet_a["v"], 1),
    "tiled_degradation_pct": round(100 * (busy["t"] - quiet_a["t"]) / quiet_a["t"], 1),
    "contention_probe": probe,
}))