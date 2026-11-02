"""P6: the driver must not submit or retry an NCU job where NCU cannot run."""

import types

from forgekernel.loop import driver


class _BoomBackend:
    def submit_gpu(self, *a, **k):
        raise AssertionError("the driver must not submit an ncu job when ncu_available is false")


def test_ncu_job_is_skipped_when_unavailable():
    ctx = types.SimpleNamespace(ncu_available=False, backend=_BoomBackend())
    assert driver._ncu_profiler(ctx, "branch", correctness=True) == {}
    assert driver._ncu_profiler(ctx, "branch", correctness=False) == {}


def test_ncu_available_defaults_to_true_on_the_context():
    fields = driver.LoopContext.__dataclass_fields__
    assert "ncu_available" in fields
    assert fields["ncu_available"].default is True