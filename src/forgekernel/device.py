"""What kind of accelerator a run's jobs use, and how that kind is selected.

A card's kernel is written for one device family: an SM architecture (``sm100``, built
by nvcc/CuTe) or an Ascend SoC (``ascend950pr``, built by the Ascend compiler). The
difference is not cosmetic -- the environment variable that selects the device, the
name of the lock that keeps two runs off it, and the arch spelling a build wants are
all different, and a harness that assumes CUDA silently hands an NPU job
``CUDA_VISIBLE_DEVICES`` and no NPU at all (or, worse, lets two runs share one device
because their lock names differ).

This module is the one place that maps a task's ``arch`` to a device kind and that
kind to its spellings, so ``submit/direct.py``, ``taskcard.py`` and ``localenv.py``
agree without any of them hard-coding "CUDA".

The physical id and the in-process logical id are kept apart on purpose: a job is
handed one physical device through its family's VISIBLE variable, and inside the
process it is always device 0 (``npu:0`` for Ascend, ``cuda:0`` for CUDA). A receipt
that records only "device 3" cannot say whether a later reading came from the same
card, so both are recorded.
"""

from __future__ import annotations

KIND_CUDA = "cuda"
KIND_NPU = "npu"
KINDS = (KIND_CUDA, KIND_NPU)
DEFAULT_KIND = KIND_CUDA

# The environment variable a job's launcher sets to give it exactly one device.
VISIBLE_ENV = {KIND_CUDA: "CUDA_VISIBLE_DEVICES", KIND_NPU: "ASCEND_RT_VISIBLE_DEVICES"}
# The lock's file-name prefix, per kind (submit/direct.py's per-device lock).
LOCK_PREFIX = {KIND_CUDA: "gpu", KIND_NPU: "npu"}
# What a process calls the one device it was given.
LOGICAL_PREFIX = {KIND_CUDA: "cuda", KIND_NPU: "npu"}
# nvcc/CuTe's arch spelling: the 'a' variants carry the arch-specific instructions the
# kernels use (H100's wgmma, Blackwell's tcgen05), so sm90 -> sm_90a, not sm_90.
CUDA_ARCH = {"sm90": "sm_90a", "sm100": "sm_100a", "sm103": "sm_103a", "sm120": "sm_120a"}
NPU_ARCH_PREFIX = "ascend"


def kind_of_arch(arch: str | None) -> str:
    """The device family a task's ``arch`` names: ``npu`` for an Ascend SoC, else CUDA.

    The test is the arch name's own prefix rather than a list of known parts, so a new
    Ascend part (``ascend910c``, ``ascend950pr``) needs no edit here -- only the KB
    tree and the arch regex in forgekernel.task.
    """
    return KIND_NPU if str(arch or "").strip().lower().startswith(NPU_ARCH_PREFIX) \
        else KIND_CUDA


def visible_env(kind: str) -> str:
    """The VISIBLE variable a job of this kind is handed."""
    return VISIBLE_ENV[kind if kind in VISIBLE_ENV else DEFAULT_KIND]


def lock_name(kind: str, physical_id: str) -> str:
    """The per-device lock's stem ("gpu0", "npu3")."""
    return f"{LOCK_PREFIX[kind if kind in LOCK_PREFIX else DEFAULT_KIND]}{physical_id}"


def logical_id(kind: str) -> str:
    """The index the job process sees: always 0, whatever physical device it was given."""
    return "0"


def logical_name(kind: str, index: str = "0") -> str:
    """``npu:0`` / ``cuda:0``: how a log line names the device inside the process."""
    return f"{LOGICAL_PREFIX[kind if kind in LOGICAL_PREFIX else DEFAULT_KIND]}:{index}"


def cuda_arch(arch: str | None) -> str:
    """nvcc/CuTe's spelling of a CUDA arch (``sm100`` -> ``sm_100a``); an unknown sm
    number gets the arch-specific suffix appended, which is what the old inline
    ``CUTE_ARCH.get(arch, arch + "a")`` did."""
    a = str(arch or "")
    return CUDA_ARCH.get(a, f"{a}a")


def describe(kind: str, physical_id: str) -> str:
    """One log line: the physical device handed out and what it is inside the process."""
    return (f"{kind.upper()} physical {physical_id} "
            f"({visible_env(kind)}={physical_id}, in-process {logical_name(kind)})")
