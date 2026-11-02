"""P8: the msprof device basis is host-free, like the CUPTI span and the graph."""

from forgekernel import baseline, timing_audit


def test_msprof_is_a_host_free_basis():
    assert baseline.BASIS_MSPROF == "msprof"
    assert "msprof" in baseline.HOST_FREE_BASES
    assert timing_audit.is_host_free_basis("msprof")


def test_a_msprof_reading_is_never_called_host_bound():
    v = timing_audit.classify(host_enqueue_ms=0.5, device_median_ms=0.05, basis="msprof")
    assert v.host_free
    assert v.measured
    assert not v.host_bound
    assert v.ratio is None


def test_a_plain_window_is_still_judged_host_bound():
    v = timing_audit.classify(host_enqueue_ms=0.5, device_median_ms=0.05, basis="plain_batch")
    assert v.host_bound and v.measured