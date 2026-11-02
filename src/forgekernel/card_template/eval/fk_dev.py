"""Device abstraction for the copied eval bench (P5: score candidates on an NPU too).

Every device fact the bench needs -- synchronize, a timing event, a stream, the RNG
device list, the L2-flush buffer, the TF32 switch, CUDA-graph capture, and the
device-time basis -- is asked of this module instead of ``torch.cuda.*`` directly.
That is what lets the same ``timing.py`` / ``precision.py`` / ``leaderboard.py``
run on a CUDA GPU and on an Ascend NPU: the CUDA-only paths are off when the kind is
``npu``, and the device window is measured with msprof kernel spans instead of CUDA
events.

Where the kind comes from: the card's ``problem.json`` ``device`` field. The writer
(forgekernel.taskcard) fills it with ``forgekernel.device.kind_of_arch(task.arch)``, so
the arch->kind rule lives in exactly one place (P3's device.py) and this file never
re-derives it. A card without the field predates NPU support and is CUDA, which is the
behaviour every archived reading was taken under.

This module is copied into the card's ``eval/`` next to the bench (see
forgekernel.taskcard EVAL_FILES / forgekernel.leaderboard.write_card), so it must not
import the forgekernel package. It imports only ``torch`` and, optionally, the
Ascend msprof helper when one is installed.
"""

from __future__ import annotations

import contextlib
import sys

# The torch module this layer was loaded against. Bound at load (a stub in a test
# harness, the real package in a job) so a module that imports torch first still drives
# the same object; the eval modules reload this file when sys.modules["torch"] changes
# identity (see their _fk_dev_module helpers), so a stubbed bench is never left using a
# stale torch.
_torch_obj = sys.modules.get("torch") or __import__("torch")

KIND_CUDA = "cuda"
KIND_NPU = "npu"
KINDS = (KIND_CUDA, KIND_NPU)
DEFAULT_KIND = KIND_CUDA

_kind = DEFAULT_KIND
_MS_CACHE = None
_MS_CACHE_READY = False


def _torch():
    """The ``torch`` module this layer was loaded against."""
    return _torch_obj


def configure(kind) -> str:
    """Set the device family for this process; ``npu`` selects the Ascend paths."""
    global _kind
    _kind = KIND_NPU if str(kind or "").strip().lower() == KIND_NPU else KIND_CUDA
    return _kind


def kind() -> str:
    return _kind


def is_npu() -> bool:
    return _kind == KIND_NPU


def is_cuda() -> bool:
    return _kind == KIND_CUDA


def device_string(kind_=None) -> str:
    """The ``torch.device`` spelling of the kind (or of an explicit kind)."""
    k = _kind if kind_ is None else kind_
    return KIND_NPU if str(k).strip().lower() == KIND_NPU else KIND_CUDA


def _backend():
    """``torch.npu`` / ``torch.cuda`` for the current kind, or None."""
    t = _torch()
    return getattr(t, "npu", None) if is_npu() else getattr(t, "cuda", None)


def available() -> bool:
    b = _backend()
    try:
        return bool(b is not None and b.is_available())
    except Exception:  # noqa: BLE001 — a stub / partial backend is simply unavailable
        return False


def synchronize(device=None) -> None:
    """Drain the device. A no-op when there is no usable backend (a CPU host)."""
    b = _backend()
    if b is None:
        return
    try:
        b.synchronize()
        return
    except TypeError:
        pass
    except Exception:  # noqa: BLE001
        return
    try:
        b.synchronize(device)
    except Exception:  # noqa: BLE001
        pass


def Event(enable_timing=True):
    """A timing event of the current backend (falls back to a wall clock with none)."""
    b = _backend()
    if b is not None and hasattr(b, "Event"):
        try:
            return b.Event(enable_timing=enable_timing)
        except TypeError:
            return b.Event()
        except Exception:  # noqa: BLE001
            pass
    return _WallClockEvent()


class _WallClockEvent:
    """A monotonic wall-clock stand-in for an unavailable device event (ms)."""

    def __init__(self):
        self._t = None

    def record(self):
        import time
        self._t = time.perf_counter()

    def elapsed_time(self, other):
        import time
        end = other._t if other._t is not None else time.perf_counter()
        return (end - self._t) * 1e3 if self._t is not None else 0.0


def current_device():
    b = _backend()
    try:
        return b.current_device() if b is not None else 0
    except Exception:  # noqa: BLE001
        return 0


def fork_rng_devices() -> list:
    """The device ids ``torch.random.fork_rng`` must save/restore (none on CPU)."""
    return [current_device()] if available() else []


def manual_seed_all(seed) -> None:
    b = _backend()
    if b is not None and hasattr(b, "manual_seed_all"):
        try:
            b.manual_seed_all(seed)
        except Exception:  # noqa: BLE001
            pass


def empty_cache() -> None:
    b = _backend()
    if b is not None and hasattr(b, "empty_cache"):
        try:
            b.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# --- CUDA-graph capture (a CUDA-only facility) ---------------------------------

def graph_available() -> bool:
    if is_npu():
        return False
    t = _torch()
    cuda = getattr(t, "cuda", None)
    return bool(cuda is not None and hasattr(cuda, "CUDAGraph") and hasattr(cuda, "graph"))


def graph_note() -> str:
    return ("Ascend NPU: CUDA-graph replay is a CUDA-only facility; use the msprof device "
            "window" if is_npu() else "")


def CUDAGraph():
    return _torch().cuda.CUDAGraph()


def Stream():
    return _torch().cuda.Stream()


def current_stream():
    return _torch().cuda.current_stream()


def stream(s):
    return _torch().cuda.stream(s)


def graph(g):
    return _torch().cuda.graph(g)


# --- L2 flush / persisting cache (CUDA-only; NPU replaced by msprof's own flush) ----

def l2_flush_buffer(device):
    """The buffer a CUDA card writes to flush L2 before a timed call; None on NPU.

    On NPU the msprof timing backend flushes the cache with its own Ascend kernel
    before every measured repeat (see ``device_ms``), so the bench must not also
    allocate a ``torch.cuda`` L2 buffer.
    """
    if not is_cuda():
        return None
    t = _torch()
    try:
        size = int(t.cuda.get_device_properties(t.device(device)).L2_cache_size) * 2
        return t.empty(max(size, 1 << 20), dtype=t.int8, device=device)
    except Exception:  # noqa: BLE001
        return None


def reset_persisting_l2() -> None:
    if not is_cuda():
        return
    try:
        _torch().cuda.cudart().cudaCtxResetPersistingL2Cache()
    except Exception:  # noqa: BLE001
        pass


# --- TF32 (a CUDA-only knob; explicitly N/A on NPU) ----------------------------

@contextlib.contextmanager
def no_tf32():
    """Run with tf32 off. On NPU there is no tf32 matrix path, so this is a no-op.

    The precision contract's native run must not inherit a tf32 flag the kernel set;
    on NPU the honesty check is the same, only the (absent) tf32 knob is skipped.
    """
    if not is_cuda():
        yield
        return
    t = _torch()
    m, c = t.backends.cuda.matmul.allow_tf32, t.backends.cudnn.allow_tf32
    t.backends.cuda.matmul.allow_tf32 = t.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        t.backends.cuda.matmul.allow_tf32, t.backends.cudnn.allow_tf32 = m, c


# --- device-time basis: msprof kernel spans on NPU, events on CUDA --------------

def _msprof():
    """``tilelang.profiler.msprof.bench_with_msprof`` when Ascend msprof is installed."""
    try:
        from tilelang.profiler.msprof import bench_with_msprof
        return bench_with_msprof
    except Exception:  # noqa: BLE001 — an optional backend
        return None


def msprof_available() -> bool:
    return is_npu() and _msprof() is not None


def _ms_cache():
    global _MS_CACHE, _MS_CACHE_READY
    if not _MS_CACHE_READY:
        _MS_CACHE_READY = True
        try:
            _MS_CACHE = _torch().empty(64 * 1024 * 1024 // 4, dtype=_torch().int32,
                                       device=device_string())
        except Exception:  # noqa: BLE001
            _MS_CACHE = None
    return _MS_CACHE


def device_ms(fn, batch: int) -> float:
    """One per-call device duration in ms from the msprof kernel spans.

    ``bench_with_msprof`` runs ``batch`` calls, each preceded by its Ascend cache-flush
    kernel, under ``torch_npu.profiler``, and returns the summed kernel duration of one
    repeat divided by the repeat count -- a host-free per-call device time. Raises when
    the optional backend is absent (callers fall back to npu events).
    """
    bench = _msprof()
    cache = _ms_cache()
    if bench is None or cache is None:
        raise RuntimeError("msprof device timing backend unavailable")
    return float(bench(fn, cache, int(batch)))


def device_timer_name() -> str:
    """The timer the device window actually uses, for the eval JSON's audit.

    CUDA keeps the historical plain_batch basis name (a batch of calls under one
    event pair), so an archived reading's basis is what it always was; only an NPU
    names its own basis (msprof / npu_events).
    """
    if is_npu():
        return "msprof" if msprof_available() else "npu_events"
    return "plain_batch"
