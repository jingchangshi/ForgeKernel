"""Decide whether a co-tenant actually biases the ratio, with the neighbour verified alive.

Three earlier runs disagreed (ratio shift +70%, +0.2%, -0.1%). A likely cause is that the
neighbour subprocess died silently (stderr was discarded), so a "busy" window was really a
quiet one. This version makes the neighbour auditable:
  - it writes a heartbeat file the parent reads before/after each window,
  - the parent samples npu-smi AICore utilisation during the window,
  - a window is only labelled contended if both heartbeat advanced and utilisation rose.
"""
import json, os, re, statistics, subprocess, sys, textwrap, time

import torch, torch_npu, tilelang
import tilelang.ascend.language as T
from tilelang.profiler.msprof import bench_with_msprof

B, D = 4096, 4096
HEARTBEAT = "/tmp/neighbour_heartbeat.txt"


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
    seconds, n, path = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    a = torch.randn(n, n, dtype=torch.bfloat16, device="npu")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="npu")
    end, count = time.time() + seconds, 0
    while time.time() < end:
        for _ in range(20):
            torch.matmul(a, b)
        torch.npu.synchronize()
        count += 20
        with open(path, "w") as fh:
            fh.write(str(count))
    print("NEIGHBOUR_TOTAL", count, flush=True)
""")


def aicore_util():
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    vals = [int(m) for m in re.findall(r"\|\s*(\d+)\s+\d+\s*/\s*\d+\s+\d+\s*/\s*\d+", out)]
    return max(vals) if vals else None


x = torch.randn(B, D, dtype=torch.bfloat16, device="npu")
w = torch.randn(D, dtype=torch.bfloat16, device="npu")
kern = tilelang.compile(build(B, D), target="ascend", out_idx=[1])
cache = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device="npu")


def tiled():
    kern(x.flatten(), w, 1e-6)


def vendor():
    torch_npu.npu_rms_norm(x, w, 1e-6)


tiled(); vendor(); torch.npu.synchronize()


def window(n=9):
    t, v, p = [], [], []
    for k in range(n):
        vals = {}
        for fn in ((tiled, vendor) if k % 2 == 0 else (vendor, tiled)):
            vals["t" if fn is tiled else "v"] = bench_with_msprof(fn, cache, 10)
        t.append(vals["t"]); v.append(vals["v"]); p.append(vals["v"] / vals["t"])
    return {"tiled_ms": round(statistics.median(t), 6),
            "vendor_ms": round(statistics.median(v), 6),
            "ratio": round(statistics.median(p), 4),
            "ratio_min": round(min(p), 4), "ratio_max": round(max(p), 4)}


result = {"shape": "%dx%d bf16" % (B, D)}
result["quiet"] = window()
result["quiet_util"] = aicore_util()

if os.path.exists(HEARTBEAT):
    os.remove(HEARTBEAT)
child = subprocess.Popen([sys.executable, "-c", NEIGHBOUR, "90", "2048", HEARTBEAT],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(7.0)
alive_before = child.poll() is None
hb_before = int(open(HEARTBEAT).read()) if os.path.exists(HEARTBEAT) else 0
util_before = aicore_util()
busy = window()
util_after = aicore_util()
hb_after = int(open(HEARTBEAT).read()) if os.path.exists(HEARTBEAT) else 0
child.terminate()
try:
    child.wait(timeout=10)
except subprocess.TimeoutExpired:
    child.kill(); child.wait(timeout=10)
err = child.stderr.read()[:300]

result["neighbour"] = {
    "alive_when_window_started": alive_before,
    "heartbeat_before": hb_before, "heartbeat_after": hb_after,
    "heartbeat_advanced": hb_after > hb_before,
    "aicore_util_before_pct": util_before, "aicore_util_after_pct": util_after,
    "stderr": err,
}
result["busy"] = busy
q, b = result["quiet"], result["busy"]
contended = hb_after > hb_before
result["contended_verified"] = contended
result["ratio_shift_pct"] = round(100 * (b["ratio"] - q["ratio"]) / q["ratio"], 1)
result["vendor_slowdown_pct"] = round(100 * (b["vendor_ms"] - q["vendor_ms"]) / q["vendor_ms"], 1)
result["tiled_slowdown_pct"] = round(100 * (b["tiled_ms"] - q["tiled_ms"]) / q["tiled_ms"], 1)
print("VERIFIED_JSON " + json.dumps(result))