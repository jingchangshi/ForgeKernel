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


def test_reference_report_distinguishes_host_free_basis(capsys):
    from forgekernel.baseline_cli import _report_host_bound
    b = baseline.RefBaseline(op="rmsnorm", ms=0.3, ms_device=0.2,
                             device_window_basis="msprof", host_enqueue_ms=0.1)
    _report_host_bound(b)
    output = capsys.readouterr().out
    assert "msprof" in output and "host-free" in output
    assert "reports no host_enqueue_ms" not in output


def test_a_plain_window_is_still_judged_host_bound():
    v = timing_audit.classify(host_enqueue_ms=0.5, device_median_ms=0.05, basis="plain_batch")
    assert v.host_bound and v.measured

def test_an_npu_msprof_reference_anchors_the_target_without_a_host_bound_refusal():
    """A reference measured on the msprof basis is host-free, so target_ms anchors on
    its device reading and the host-bound blocker must not fire even though the plain
    enqueue ratio looks high (0.77)."""
    from forgekernel import baseline
    b = baseline.RefBaseline(op="rmsnorm", role=baseline.ROLE_ARCHITECTURE, ms=0.30,
                             ms_device=0.26, device_window_basis="msprof",
                             host_enqueue_ms=0.20, host_bound_ratio=0.77)
    assert b.device_anchor_basis == "msprof"
    assert b.device_anchor_ms == 0.26
    assert b.host_bound.host_bound is False
    assert b.target_ms(1.0) == 0.26
