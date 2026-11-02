"""The vendor task measures the intended operator and returns only its output."""
import importlib.util
import sys
import types
from pathlib import Path


def test_vendor_reference_uses_fixed_epsilon_and_output(monkeypatch):
    calls = []
    module = types.ModuleType("torch_npu")
    module.npu_rms_norm = lambda x, w, eps: (calls.append((x, w, eps)) or "output", "rstd")
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    path = Path(__file__).resolve().parents[1] / "tasks/rmsnorm-ascend-vendor/reference/run.py"
    spec = importlib.util.spec_from_file_location("vendor_rmsnorm_reference", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    x, w = object(), object()
    assert ref.run(x, w) == "output"
    assert calls == [(x, w, 1e-6)]
