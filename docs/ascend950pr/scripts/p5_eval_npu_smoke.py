"""P5 minimal evidence: the device-neutral eval bench scores one candidate per
shape on an Ascend NPU, and the float64 distance is computed on the host.

Runs <120s on npu:1 (ASCEND_RT_VISIBLE_DEVICES=1). It drives the COPIED template
modules (src/forgekernel/card_template/eval/{fk_dev,timing,precision}.py) exactly as a
job would: problem device = npu, candidate = a torch_npu RMSNorm on real bf16 tensors.
No leaderboard board, no NCU, no CUDA.

Prints one JSON object prefixed P5_SMOKE.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/s00653124/workspace/ForgeKernel")
EVAL = REPO / "src" / "forgekernel" / "card_template" / "eval"
sys.path.insert(0, str(EVAL))

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

import fk_dev
import timing
import precision

fk_dev.configure("npu")

# Keep the whole smoke well under 120 s: smaller windows than a production job.
timing.WARMUP_ITERS = 2
timing.E2E_ITERS = 5
timing.DEVICE_BATCH = 10
timing.DEVICE_REPS = 2
timing.HOST_ENQUEUE_ITERS = 5

out = {"visible_env": "ASCEND_RT_VISIBLE_DEVICES",
       "device": "npu:0 (the one physical device the launcher hands this job)",
       "torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", "?"),
       "msprof_available": fk_dev.msprof_available(), "shapes": []}


def rms_fn(x, w):
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    return ((xf * inv) * w.float()).to(x.dtype)


def timed_call(x, w):
    def call():
        return rms_fn(x, w)
    return call


t0 = time.time()

# 1) a direct device-time reading, so a broken msprof backend fails loudly here
try:
    probe_x = torch.randn(64, 4096, dtype=torch.bfloat16, device="npu")
    probe_w = torch.randn(4096, dtype=torch.bfloat16, device="npu")
    out["msprof_probe_ms"] = fk_dev.device_ms(timed_call(probe_x, probe_w), 5)
    out["msprof_probe_ok"] = True
except Exception as exc:  # noqa: BLE001
    out["msprof_probe_ok"] = False
    out["msprof_probe_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"

# 2) per-shape performance score through timing.py's workload windows
for name, (M, D) in (("decode_64x4096", (64, 4096)), ("prefill_4096x4096", (4096, 4096))):
    x = torch.randn(M, D, dtype=torch.bfloat16, device="npu")
    w = torch.randn(D, dtype=torch.bfloat16, device="npu")
    fn = timed_call(x, w)
    fn()
    torch.npu.synchronize()
    try:
        result = timing.measure_windows_workloads([(name, [fn], None)], burn_in_s=0.0)
        block = timing.workloads_performance(result)
        out["shapes"].append({
            "shape": name, "M": M, "D": D,
            "device_median_ms": block["device_median_ms"],
            "e2e_median_ms": block["bwd_median_ms"],
            "device_window_basis": timing.device_timer_basis(),
        })
    except Exception as exc:  # noqa: BLE001
        out["shapes"].append({"shape": name, "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
    del x, w

# 3) the float64 distance: the NPU .double() silent downgrade vs the host result
xf = torch.randn(256, 4096, dtype=torch.float32, device="npu")
out["npu_double_dtype"] = str(xf.double().dtype)          # documented silent downgrade
truth64 = xf.to("cpu").to(torch.float64) * (1.0 + 1e-6)
out["rel_l2"] = precision.rel_l2(xf.to("cpu"), truth64)  # the bench's real distance
out["rel_l2_finite"] = bool(0.0 <= out["rel_l2"] < 1e9)
out["rel_l2_expected_scale"] = "~1e-6 (float32 rounding of a float64 truth)"

out["elapsed_s"] = round(time.time() - t0, 2)
print("P5_SMOKE " + json.dumps(out))
