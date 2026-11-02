"""How much does one canary measurement cost? Answer it before designing the witness.

accept-verify can measure a canary kernel inside each window, but a single microsecond-scale
observation can miss an intermittent neighbour. The fix would be N canaries per window, and
the question is what that costs. Guessing at the cost would set the protocol on a fiction, so
measure it: the canary is a small vendor operator timed through msprof, which is what the
real one would be.
"""
import json, statistics, torch, torch_npu
from tilelang.profiler.msprof import bench_with_msprof

B, D = 64, 4096
x = torch.randn(B, D, dtype=torch.bfloat16, device="npu")
w = torch.randn(D, dtype=torch.bfloat16, device="npu")
cache = torch.empty(8 * 1024 * 1024 // 4, dtype=torch.int32, device="npu")


def canary():
    torch_npu.npu_rms_norm(x, w, 1e-6)


canary()
torch.npu.synchronize()

# One canary reading as the protocol would take it.
import time
calls = []
for _ in range(12):
    t0 = time.perf_counter()
    ms = bench_with_msprof(canary, cache, 3)
    calls.append({"wallSeconds": round(time.perf_counter() - t0, 4),
                  "deviceMs": round(ms, 6) if isinstance(ms, (int, float)) else None})

wall = [c["wallSeconds"] for c in calls]
out = {
  "shape": "%dx%d bf16 (vendor op as canary)" % (B, D),
  "samples": calls,
  "wallCostPerCanary_ms": {"min": round(min(wall) * 1000, 1), "median": round(statistics.median(wall) * 1000, 1),
                           "max": round(max(wall) * 1000, 1)},
  "deviceTimeBeingJudged_ms": 0.004,
  "ratio_canary_cost_to_device_time": round(statistics.median(wall) * 1000 / 0.004, 1),
}
# How many canaries fit in a window without dominating it?
med = statistics.median(wall) * 1000
out["canaries_per_second_of_window"] = round(1000 / med, 2)
out["cost_of_5_canaries_ms"] = round(5 * med, 1)
out["cost_of_10_canaries_ms"] = round(10 * med, 1)
print("CANARY_JSON " + json.dumps(out))