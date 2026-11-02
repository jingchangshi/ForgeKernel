"""The NPU branch of the copied eval bench (P5): the device window, CUDA-graph
refusal, L2/TF32 no-ops and the CPU-float64 distance all run with NO torch.cuda.

A stub torch with a fake ``torch.npu`` and deliberately no ``torch.cuda`` proves the
NPU code path never reaches for a CUDA API; the real-torch cases pin the CPU-float64
distance and the problem.json -> device wiring. The CUDA behaviour itself is guarded by
test_template_timing.py / test_template_paired_timing.py (unchanged), which run the same
code with a CUDA stub.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

EVAL = (Path(__file__).resolve().parents[1]
        / "src" / "forgekernel" / "card_template" / "eval")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGpuNPU:
    """An NPU whose per-call cost advances a device clock the fake events read."""

    def __init__(self, device_ms: float = 1.0):
        self.device_ms = device_ms
        self.clock_ms = 0.0
        self.calls = 0

    def call(self):
        self.calls += 1
        self.clock_ms += self.device_ms


def _fake_torch_npu(gpu: FakeGpuNPU):
    """A stub torch with torch.npu ONLY -- no torch.cuda attribute at all."""
    t = types.ModuleType("torch")
    npu = types.ModuleType("torch.npu")

    class Event:
        def __init__(self, enable_timing=False):
            self.t = 0.0

        def record(self):
            self.t = gpu.clock_ms

        def elapsed_time(self, other):
            return other.t - self.t

    npu.Event = Event
    npu.synchronize = lambda *a, **k: None
    npu.is_available = lambda: True
    npu.current_device = lambda: 0
    npu.manual_seed_all = lambda *a, **k: None
    npu.empty_cache = lambda *a, **k: None
    t.npu = npu
    # NO t.cuda on purpose: any torch.cuda reach on the NPU path is an AttributeError.
    t.equal = lambda a, b: getattr(a, "data", a) == getattr(b, "data", b)
    return t


def load_timing(torch_mod):
    """Import timing.py fresh against torch_mod, sharing a fresh fk_dev."""
    sys.modules.pop("fk_dev", None)
    saved = sys.modules.get("torch")
    sys.modules["torch"] = torch_mod
    try:
        return _load("_npu_timing", EVAL / "timing.py")
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def test_timing_npu_needs_no_torch_cuda():
    gpu = FakeGpuNPU(device_ms=1.0)
    t = load_timing(_fake_torch_npu(gpu))
    t.fk_dev.configure("npu")
    # kill the optional msprof backend so this exercises the numpy-event fallback path
    t.fk_dev._msprof = lambda: None
    assert not hasattr(t.fk_dev._torch(), "cuda")
    out = t.measure_windows(gpu.call)
    assert out["device_median_ms"] == pytest.approx(1.0)
    assert out["median_ms"] == pytest.approx(1.0)
    assert out["host_bound_ratio"] < t.HOST_BOUND_RATIO_WARN


def test_msprof_device_window_is_preferred_on_npu(monkeypatch):
    gpu = FakeGpuNPU(device_ms=9.0)          # the event clock says 9 ms/call ...
    t = load_timing(_fake_torch_npu(gpu))
    t.fk_dev.configure("npu")
    monkeypatch.setattr(t.fk_dev, "msprof_available", lambda: True)
    monkeypatch.setattr(t.fk_dev, "device_ms", lambda fn, batch: 0.25)  # ... msprof says 0.25
    samples = t.time_device(gpu.call, warmup=1, batch=3, reps=2)
    assert samples == [0.25, 0.25]
    assert t.device_timer_basis() == "msprof"


def test_graph_window_is_refused_on_npu_not_silently_faked():
    t = load_timing(_fake_torch_npu(FakeGpuNPU()))
    t.fk_dev.configure("npu")
    out = t.time_device_graph(lambda: None, reference_ms=1.0)
    assert out["available"] is False and out["device_median_ms"] is None
    assert "NPU" in out["reason"]


def test_l2_flush_and_tf32_are_noops_on_npu():
    t = load_timing(_fake_torch_npu(FakeGpuNPU()))
    t.fk_dev.configure("npu")
    assert t.fk_dev.l2_flush_buffer("npu") is None
    with t.fk_dev.no_tf32():                 # must not touch torch.backends.cuda
        pass
    assert t.device_timer_basis() in ("msprof", "npu_events")


def test_problem_device_kind_defaults_to_cuda():
    lb = _load("_npu_lb", EVAL / "leaderboard.py")
    assert lb.problem_device_kind({"device": "npu"}) == "npu"
    assert lb.problem_device_kind({}) == "cuda"
    assert lb.problem_device_kind({"device": None}) == "cuda"


def test_cuda_basis_label_is_unchanged():
    """A CUDA card's device basis stays the historical plain_batch (no drift)."""
    t = _load("_cuda_timing", EVAL / "timing.py")
    assert t.device_timer_basis() == "plain_batch"


def test_rel_l2_moves_to_host_and_uses_true_float64():
    P = _load("_npu_precision", EVAL / "precision.py")
    truth = torch.randn(4096, dtype=torch.float64)
    out = truth.float()                      # a float32 rounding of the truth
    got = P.rel_l2(out, truth)
    want = float(torch.linalg.vector_norm(out.double() - truth)
                 / torch.linalg.vector_norm(truth))
    assert got == pytest.approx(want, rel=1e-12)
    assert got > 0.0


def test_task_error_computes_on_host_float64():
    lb = _load("_npu_lb3", EVAL / "leaderboard.py")
    ref = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    out = ref.clone()
    assert lb.task_error(out, ref, "float32") == (0.0, None)
    off = ref.clone()
    off[0] += 1e-3
    err, why = lb.task_error(off, ref, "float32")
    assert why is None and err > 0.0


def test_reference_output_preserves_worst_host_bound_ratio(monkeypatch):
    import json
    from forgekernel.baseline import RefBaseline
    from forgekernel.baseline_cli import parse_ref_stdout

    lb = _load("_reference_audit_lb", EVAL / "leaderboard.py")
    blocks = [
        {"name": "a", "bwd_median_ms": 1.0, "device_median_ms": 1.0,
         "host_enqueue_ms": 0.04, "host_bound_ratio": 0.04},
        {"name": "b", "bwd_median_ms": 1.0, "device_median_ms": 1.0,
         "host_enqueue_ms": 0.88, "host_bound_ratio": 0.88},
    ]
    views = [{"_wl_name": b["name"], "workload": {"axes": {}}} for b in blocks]
    monkeypatch.setattr(lb, "_task_shapes", lambda problem: views)
    monkeypatch.setattr(lb, "_correctness", lambda *args: ({"pass": True}, []))
    monkeypatch.setattr(lb, "_task_side", lambda v, *args: next(b for b in blocks if b["name"] == v["_wl_name"]))
    monkeypatch.setattr(lb, "_task_aggregate", lambda problem, items: {
        "bwd_median_ms": 1.0, "device_median_ms": 1.0,
        "host_enqueue_ms": (0.04 * 0.88) ** 0.5,
        "host_bound_ratio": 0.88, "aggregate": "geomean",
        "n_workloads": 2, "workloads": items,
    })
    monkeypatch.setattr(lb, "clock_state", lambda: "unlocked")
    emitted = []
    monkeypatch.setattr(lb, "_emit", emitted.append)
    assert lb._task_ref_alone({}, object(), False, "cuda") == 0
    result = parse_ref_stdout(json.dumps(emitted[0]))
    assert result["host_bound_ratio"] == pytest.approx(0.88)
    assert [w["host_bound_ratio"] for w in emitted[0]["workloads"]] == [0.04, 0.88]
    baseline = RefBaseline(op="rmsnorm", ms=result["ms"],
                           ms_device=result["ms_device"],
                           host_enqueue_ms=result["host_enqueue_ms"],
                           host_bound_ratio=result["host_bound_ratio"])
    assert baseline.host_bound.host_bound


def test_card_problem_carries_the_device_derived_from_arch():
    from forgekernel import taskcard

    def task(arch):
        return types.SimpleNamespace(
            name="t", language="tilelang-ascend", arch=arch,
            shapes=[types.SimpleNamespace(name="s0", axes={"M": 64}, inputs=[])],
            target=types.SimpleNamespace(timing="device"))

    check = types.SimpleNamespace(definition={"inputs": {}, "outputs": {}},
                                  groups={}, judge_src="")
    assert taskcard.card_problem(task("ascend950pr"), check)["device"] == "npu"
    assert taskcard.card_problem(task("sm100"), check)["device"] == "cuda"
