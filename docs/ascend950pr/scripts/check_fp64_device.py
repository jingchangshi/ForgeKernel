"""Verify the P5 claims about device-side float64 on Ascend, rather than assuming them.

P5 asserts that moving eval onto the NPU requires relocating float64 math to CPU. That is
only true if float64 actually fails on the device. This checks each primitive the eval code
uses: float64 tensors, .double(), .mean(), torch.equal, and item() on a float64 result.
"""
import json, torch, torch_npu

out = {"device": "npu:0", "checks": []}


def check(name, fn):
    try:
        out["checks"].append({"op": name, "ok": True, "value": fn()})
    except Exception as exc:
        out["checks"].append({"op": name, "ok": False, "error": type(exc).__name__ + ": " + str(exc)[:180]})


check("bfloat16_alloc", lambda: str(torch.zeros(4, 4, dtype=torch.bfloat16, device="npu").dtype))
check("float32_alloc", lambda: str(torch.zeros(4, 4, dtype=torch.float32, device="npu").dtype))
check("float64_alloc", lambda: str(torch.zeros(4, 4, dtype=torch.float64, device="npu").dtype))


def to_float64():
    x = torch.randn(8, 8, dtype=torch.float32, device="npu")
    return str(x.double().dtype)


check("cast_to_float64", to_float64)


def float64_mean():
    x = torch.randn(8, 8, dtype=torch.float32, device="npu").double()
    return round(float(x.mean().item()), 8)


check("float64_mean_item", float64_mean)


def precision_rel_l2_style():
    """The shape of precision.rel_l2: distance in float64, then a scalar."""
    a = torch.randn(64, 128, dtype=torch.float32, device="npu")
    b = a + 1e-4
    diff = (a.double() - b.double()).flatten()
    return round(float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b.double())).item()), 10)


check("rel_l2_in_float64_on_device", precision_rel_l2_style)


def mean_pow_chain():
    x = torch.randn(64, 128, dtype=torch.bfloat16, device="npu")
    rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return {"rstd_dtype": str(rstd.dtype), "finite": bool(torch.isfinite(rstd).all())}


check("fp32_stats_from_bf16", mean_pow_chain)

if all(c["ok"] for c in out["checks"]):
    out["verdict"] = "float64 works on this device, so P5 relocation is a portability default, not a hard requirement"
else:
    out["verdict"] = "some float64 paths fail on device, confirming P5 relocation is required"
print("FP64_JSON " + json.dumps(out))