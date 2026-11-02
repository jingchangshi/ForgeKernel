"""Quantify the information loss in the reference-alone emission.

Claim under test: dropping per-workload host_bound_ratio from the ref-alone JSON lets the
archive re-derive a ratio that passes the audit even when a workload is almost entirely
host-bound.

The identity that matters: geomean(a_i / b_i) == geomean(a_i) / geomean(b_i). So an archive
that divides the aggregated host enqueue by the aggregated device time computes the GEOMEAN
of the per-workload ratios - never the maximum. timing.py deliberately emits the MAXIMUM
(`host_bound_ratio: round(max(hb), 3)`), so this re-derivation silently changes the statistic.
"""
import json, math

WARN = 0.5  # HOST_BOUND_RATIO_WARN in both timing.py and timing_audit.py

cases = {
    "mixed_quiet_and_stalled": {"host_enqueue_ms": [0.004, 0.088], "device_median_ms": [0.10, 0.10]},
    "one_stalled_workload": {"host_enqueue_ms": [0.005, 0.088], "device_median_ms": [0.10, 0.10]},
    "two_workloads_88_and_4": {"host_enqueue_ms": [0.088, 0.004], "device_median_ms": [0.10, 0.10]},
}


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


out = {"warn_threshold": WARN, "cases": {}}
for name, c in cases.items():
    per = [h / d for h, d in zip(c["host_enqueue_ms"], c["device_median_ms"])]
    max_per = max(per)
    geo_per = geomean(per)
    agg = geomean(c["host_enqueue_ms"]) / geomean(c["device_median_ms"])
    out["cases"][name] = {
        "per_workload_ratios": [round(p, 4) for p in per],
        "max_per_workload (what timing.py emits)": round(max_per, 4),
        "geomean_per_workload": round(geo_per, 4),
        "ratio_of_geomeans (what an archive re-derives)": round(agg, 4),
        "identity_holds": abs(agg - geo_per) < 1e-12,
        "audit_on_max": "BLOCKED (host-bound)" if max_per >= WARN else "passes",
        "audit_on_rederived": "BLOCKED (host-bound)" if agg >= WARN else "passes",
        "verdict_differs": (max_per >= WARN) != (agg >= WARN),
    }

out["finding"] = ("For a workload mix where one workload is host-bound and another is quiet, the maximum "
                  "reports host-bound while the re-derived geomean reports fine. The archived reference "
                  "therefore can pass a device-timing audit while one of its workloads is almost entirely "
                  "host time - which is exactly the case a device-gated baseline must refuse.")
print("HOSTBOUND_JSON " + json.dumps(out, indent=None))