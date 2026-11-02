"""P4: the job entry is rendered per device family, but its contract is not.

On the Ascend NPU the NVIDIA clock lock and cupti-python do not exist, so the entry
must not attempt them; the marker, the timeout and the task dispatch are what the loop
relies on and must be identical on both device families.
"""

from forgekernel import leaderboard as lbmod

_BASE = dict(name="rmsnorm", suite="forgekernel", arch="sm_100a",
             clock="", mem_clock="", require_lock="0", clock_note="n")


def _render(install_cupti):
    return lbmod.GPU_ENTRY_TEMPLATE.format(install_cupti=install_cupti, **_BASE)


def test_cuda_installs_cupti_and_the_npu_does_not():
    cuda = _render("1")
    npu = _render("0")
    assert "LB_INSTALL_CUPTI:-1" in cuda
    assert "LB_INSTALL_CUPTI:-0" in npu
    assert "LB_INSTALL_CUPTI:-1" not in npu


def test_the_marker_timeout_and_dispatch_are_identical_across_devices():
    cuda = _render("1")
    npu = _render("0")
    for line in (
        "[gpu_entry.sh] task=$TASK finished",
        "[gpu_entry.sh] timeout=${TASK_TIMEOUT}s",
        "[gpu_entry.sh] KILLED: task exceeded ${TASK_TIMEOUT}s timeout",
        "correctness) run_with_timeout python3 eval/bench.py --mode correctness",
        "perf|all)    run_with_timeout python3 eval/bench.py --mode all",
        "cmd)         run_with_timeout bash -c",
    ):
        assert line in cuda, line
        assert line in npu, line


def test_install_cupti_is_the_only_rendered_difference():
    cuda = _render("1").splitlines()
    npu = _render("0").splitlines()
    assert len(cuda) == len(npu)
    differing = [(a, b) for a, b in zip(cuda, npu) if a != b]
    assert len(differing) == 1
    assert "LB_INSTALL_CUPTI" in differing[0][0]