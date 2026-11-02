"""Quantify the silent float64 downgrade, because it can fake precision in the judge.

`.double()` on this device returns float32. A judge that computes in "float64" therefore
gets float32 accuracy while believing it has headroom, which can hide a real error or
report a false one. This measures how much the downgrade actually costs.
"""
import json, torch, torch_npu

out = {"checks": []}


def check(name, fn):
    try:
        out["checks"].append({"op": name, "ok": True, "value": fn()})
    except Exception as exc:
        out["checks"].append({"op": name, "ok": False, "error": type(exc).__name__ + ": " + str(exc)[:160]})


out["dtype_reality"] = {
    "alloc_float64": str(torch.zeros(2, 2, dtype=torch.float64, device="npu").dtype),
    "cast_double_of_float32": str(torch.zeros(2, 2, dtype=torch.float32, device="npu").double().dtype),
}


def on_device_vs_cpu():
    """Same computation, device (downgraded) vs CPU (true float64)."""
    torch.manual_seed(11)
    a = torch.randn(256, 256, dtype=torch.float32)
    b = torch.randn(256, 256, dtype=torch.float32)
    a_npu, b_npu = a.to("npu"), b.to("npu")
    device_norm = torch.linalg.vector_norm((a_npu.double() - b_npu.double()).flatten()).item()
    cpu_norm = torch.linalg.vector_norm((a.double() - b.double()).flatten()).item()
    return {"device_value": round(float(device_norm), 8),
            "cpu_float64_value": round(float(cpu_norm), 8),
            "abs_diff": round(abs(float(device_norm) - float(cpu_norm)), 10),
            "rel_diff": round(abs(float(device_norm) - float(cpu_norm)) / float(cpu_norm), 10)}


check("device_double_vs_cpu_double", on_device_vs_cpu)


def cancellation_case():
    """Catastrophic cancellation: where float32 vs float64 genuinely diverge."""
    x = torch.full((1, 1024), 1e4, dtype=torch.float32)
    x[0, :512] = 1e4 + 1
    x[0, 512:] = -1e4
    cpu = (x.double().sum()).item()
    npu = (x.to("npu").double().sum()).item()
    return {"exact_expected": 512.0,
            "cpu_float64": round(float(cpu), 6),
            "device_double": round(float(npu), 6),
            "device_error": round(abs(float(npu) - 512.0), 6)}


check("cancellation_sensitive_sum", cancellation_case)

out["conclusion"] = ("The judge must not rely on .double() on this device: it silently yields float32, "
                     "so a distance computed there has float32 accuracy. Precision judging must run on CPU in "
                     "true float64 after copying the tensors back, and the copy-back cost must be excluded from device timing.")
print("FP64_DOWNGRADE_JSON " + json.dumps(out))