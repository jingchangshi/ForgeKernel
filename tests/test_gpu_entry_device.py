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


def test_npu_entry_does_not_touch_nvidia_clocks_or_install_cupti(tmp_path):
    import os
    import subprocess

    marker = tmp_path / "smi-called"
    fake = tmp_path / "nvidia-smi"
    fake.write_text('#!/bin/sh\ntouch "$SMI_MARKER"\n')
    fake.chmod(0o755)
    entry = tmp_path / "gpu_entry.sh"
    entry.write_text(_render("0"))
    env = dict(os.environ, LB_NVIDIA_SMI=str(fake), SMI_MARKER=str(marker),
               LB_SM_CLOCK_MHZ="1500", LB_REQUIRE_CLOCK_LOCK="1",
               LB_INSTALL_CUPTI="1", SOL_ENV=str(tmp_path / "absent"))
    result = subprocess.run(["bash", str(entry), "cmd", "true"], env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()
    assert "task=cmd finished" in result.stdout


def test_npu_ncu_task_does_not_launch_ncu(tmp_path):
    import os
    import subprocess

    marker = tmp_path / "ncu-called"
    fake = tmp_path / "ncu"
    fake.write_text('#!/bin/sh\ntouch "$NCU_MARKER"\n')
    fake.chmod(0o755)
    entry = tmp_path / "gpu_entry.sh"
    entry.write_text(_render("0"))
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
               NCU_MARKER=str(marker), SOL_ENV=str(tmp_path / "absent"))
    result = subprocess.run(["bash", str(entry), "ncu"], env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()
    assert "NCU unavailable for this device" in result.stdout