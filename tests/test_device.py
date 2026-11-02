"""The device abstraction (forgekernel/device.py) and the three places that used to
assume CUDA: the job environment, the per-device lock, and local.env's FK_GPU wording.

The failure mode this pins: a harness that hard-codes ``CUDA_VISIBLE_DEVICES`` hands an
Ascend job no NPU (or whatever the machine's default is, which may be someone else's
card) and puts it under a lock nobody else takes, so two runs on one NPU never serialise
and the receipt cannot say which physical device was used.
"""

from __future__ import annotations

import threading

import pytest

from forgekernel import device
from forgekernel.localenv import load_machine
from forgekernel.submit import direct
from forgekernel.submit.config import SubmitConfig, load_config


# -- the mapping ---------------------------------------------------------------------------

def test_an_ascend_arch_is_an_npu_and_everything_else_is_cuda():
    assert device.kind_of_arch("ascend950pr") == device.KIND_NPU
    assert device.kind_of_arch("ascend910c") == device.KIND_NPU
    assert device.kind_of_arch("ascend") == device.KIND_NPU
    for cuda in ("sm90", "sm100", "sm103", "sm120", "", None):
        assert device.kind_of_arch(cuda) == device.KIND_CUDA, cuda
    # case and stray whitespace do not change the family
    assert device.kind_of_arch("  Ascend950PR ") == device.KIND_NPU


def test_each_family_has_its_own_visible_variable_lock_and_logical_name():
    assert device.visible_env(device.KIND_CUDA) == "CUDA_VISIBLE_DEVICES"
    assert device.visible_env(device.KIND_NPU) == "ASCEND_RT_VISIBLE_DEVICES"
    assert device.lock_name(device.KIND_CUDA, "3") == "gpu3"
    assert device.lock_name(device.KIND_NPU, "3") == "npu3"
    # the lock names differ, so a CUDA run and an NPU run on the same integer never
    # think they are waiting on the same device
    assert device.lock_name(device.KIND_CUDA, "0") != device.lock_name(device.KIND_NPU, "0")
    assert device.logical_name(device.KIND_NPU) == "npu:0"
    assert device.logical_name(device.KIND_CUDA) == "cuda:0"
    # in-process the job always sees device 0, whatever physical id it was handed
    assert device.logical_id(device.KIND_NPU) == "0"
    assert device.logical_id(device.KIND_CUDA) == "0"


def test_the_physical_id_and_the_in_process_name_are_both_reported():
    line = device.describe(device.KIND_NPU, "7")
    assert "7" in line and "npu:0" in line and "ASCEND_RT_VISIBLE_DEVICES" in line
    assert device.describe(device.KIND_CUDA, "2").count("cuda:0") == 1


def test_the_cuda_arch_spelling_is_the_arch_specific_one():
    """sm90 -> sm_90a, not sm_90: the 'a' variants are what the kernels' wgmma/tcgen05
    instructions need, and this is where that spelling now lives."""
    assert device.cuda_arch("sm90") == "sm_90a"
    assert device.cuda_arch("sm100") == "sm_100a"
    assert device.cuda_arch("sm103") == "sm_103a"
    assert device.cuda_arch("sm120") == "sm_120a"
    # an unknown sm number keeps the old fallback rather than crashing mid-launch
    assert device.cuda_arch("sm105") == "sm105a"


# -- the job environment -------------------------------------------------------------------

def _cfg(tmp_path, kind, gpu="2"):
    d = tmp_path / "operators" / "op"
    d.mkdir(parents=True, exist_ok=True)
    (d / "submit.toml").write_text(
        '[job]\ntimeout = 60\n[repo]\ndelivery = "local"\nbranch = "main"\n'
        f'[exec]\ngpu = "{gpu}"\nkind = "{kind}"\n[env]\nA = "1"\n')
    return load_config("op", base_dir=tmp_path)


def test_an_npu_job_is_handed_the_ascend_variable_and_not_the_cuda_one(tmp_path):
    cfg = _cfg(tmp_path, "npu")
    env = direct._job_env(cfg)
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "2"
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["A"] == "1" and env["TASK_TIMEOUT"] == "60"
    # ... and a CUDA job is exactly what it always was
    cu = direct._job_env(_cfg(tmp_path, "cuda", gpu="5"))
    assert cu["CUDA_VISIBLE_DEVICES"] == "5"
    assert "ASCEND_RT_VISIBLE_DEVICES" not in cu


def test_a_card_written_before_the_kind_key_is_a_cuda_card(tmp_path):
    d = tmp_path / "operators" / "op"
    d.mkdir(parents=True)
    (d / "submit.toml").write_text(
        '[repo]\ndelivery = "local"\nbranch = "main"\n[exec]\ngpu = "0"\n')
    cfg = load_config("op", base_dir=tmp_path)
    assert cfg.exec_kind == device.KIND_CUDA
    assert "CUDA_VISIBLE_DEVICES" in direct._job_env(cfg)


def test_an_unknown_device_kind_is_refused_not_read_as_cuda(tmp_path):
    d = tmp_path / "operators" / "op"
    d.mkdir(parents=True)
    (d / "submit.toml").write_text(
        '[repo]\ndelivery = "local"\nbranch = "main"\n[exec]\ngpu = "0"\nkind = "gpu"\n')
    with pytest.raises(ValueError, match=r"invalid \[exec\] kind"):
        load_config("op", base_dir=tmp_path)


# -- the lock ------------------------------------------------------------------------------

def test_the_lock_name_follows_the_device_family(tmp_path, monkeypatch):
    monkeypatch.setenv(direct.LOCK_DIR_ENV, str(tmp_path / "locks"))
    with direct.gpu_lock("0", kind=device.KIND_NPU):
        assert (tmp_path / "locks" / "npu0.lock").exists()
    with direct.gpu_lock("0", kind=device.KIND_CUDA):
        assert (tmp_path / "locks" / "gpu0.lock").exists()


def test_an_npu_lock_does_not_block_a_cuda_lock_on_the_same_number(tmp_path, monkeypatch):
    """Physical Ascend device 0 and CUDA device 0 are different devices on a machine that
    has both; one run must not wait on the other."""
    monkeypatch.setenv(direct.LOCK_DIR_ENV, str(tmp_path / "locks"))
    got = threading.Event()

    def _cuda():
        with direct.gpu_lock("0", kind=device.KIND_CUDA):
            got.set()

    with direct.gpu_lock("0", kind=device.KIND_NPU):
        t = threading.Thread(target=_cuda)
        t.start()
        assert got.wait(timeout=2.0), "a CUDA lock waited on an NPU lock"
    t.join(timeout=5)


def test_two_npu_runs_on_one_device_still_serialise(tmp_path, monkeypatch):
    monkeypatch.setenv(direct.LOCK_DIR_ENV, str(tmp_path / "locks"))
    calls: list[str] = []

    class _Log:
        def write(self, s):
            calls.append(s)

        def flush(self):
            pass

    got = threading.Event()

    def _second():
        with direct.gpu_lock("1", _Log(), poll_s=0.05, kind=device.KIND_NPU):
            got.set()

    with direct.gpu_lock("1", kind=device.KIND_NPU):
        t = threading.Thread(target=_second)
        t.start()
        assert not got.wait(timeout=0.3)
    assert got.wait(timeout=5)
    assert any("NPU 1 is in use by another forgekernel run" in c for c in calls)


# -- local.env -----------------------------------------------------------------------------

def test_an_npu_device_id_is_a_valid_fk_gpu(tmp_path):
    """"FK_GPU must be a CUDA device index" was wrong twice over: the key also names an
    Ascend device, and the value is an index either way."""
    m = load_machine(tmp_path, environ={"FK_GPU": "3"})
    assert m.gpu == "3"
    assert m.describe() == "this machine (device 3)"
    m = load_machine(tmp_path, environ={"FK_GPU": "3", "FK_DELIVERY": "ssh",
                                        "REMOTE_HOST": "box", "REMOTE_WORKDIR": "/d/fk"})
    assert m.describe() == "ssh box:/d/fk (device 3)"


def test_a_non_numeric_device_is_refused_without_naming_cuda(tmp_path):
    with pytest.raises(Exception) as e:
        load_machine(tmp_path, environ={"FK_GPU": "npu:0"})
    text = str(e.value)
    assert "FK_GPU" in text and "device index" in text
    assert "CUDA device index" not in text
