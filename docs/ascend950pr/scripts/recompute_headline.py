"""Independently recompute the two headline ratios from the raw per-window samples.

This deliberately does not reuse the verdict stored in accept-genuine.json: if the
report quotes a number, that number should be reproducible from the samples, not
trusted because a JSON file already contains a summary of it.
"""
from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import json, math, statistics, random
from pathlib import Path
P = Path(str(PLAN) + "")
acc = json.load(open(P / "artifacts/accept-genuine.json"))

def boot_ci(xs, n=4000, seed=7):
    rnd = random.Random(seed)
    meds = sorted(statistics.median(rnd.choices(xs, k=len(xs))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]

print("%-11s %-8s %-22s %-22s %-9s" % ("shape", "windows", "recomputed point+CI", "file-reported", "agree"))
for shape, v in acc["timing"].items():
    ws = v["windows"]
    per = [w["vendor_ms"] / w["candidate_ms"] for w in ws]
    pt = statistics.median(per)
    lo, hi = boot_ci(per)
    fpt = v["pointSpeedup"]
    flo, fhi = v["interval95"]
    # The point estimate must reproduce to within the precision the report quotes (4 dp);
    # the bootstrap interval depends on the resampling seed, so it is compared with a
    # tolerance rather than exactly. Anything beyond that would be a real discrepancy.
    agree = abs(pt - fpt) < 5e-4 and abs(lo - flo) < 5e-3 and abs(hi - fhi) < 5e-3
    print("%-11s %-8d %-22s %-22s %-9s" % (
        shape, len(per),
        "%.4f [%.4f,%.4f]" % (pt, lo, hi),
        "%.4f [%.4f,%.4f]" % (fpt, flo, fhi),
        agree))
    # is the verdict implied by the samples themselves?
    if hi < 1.0:
        implied = "FAIL (upper < 1.0)"
    elif lo >= 0.98 and pt >= 1.0:
        implied = "MATCH/EXCEED"
    else:
        implied = "RETEST (interval spans threshold)"
    print("%-11s %s" % ("", "verdict implied by raw samples: " + implied))

print()
print("Cross-run reproducibility of the prefill ratio (same artifact, three independent measurements):")
print("  probe (cold, do_bench)      0.3148")
print("  paired_bench (15 pairs x20) 0.3124")
print("  accept (10 paired windows)  0.3224")
vals = [0.3148, 0.3124, 0.3224]
print("  spread: %.1f%% of the median (%.4f)" % (100 * (max(vals) - min(vals)) / statistics.median(vals), statistics.median(vals)))
print()
print("Bandwidth arithmetic, recomputed from the measured times:")
bytes_moved = 8192 * 4096 * 2
for label, us in (("vendor", 51.43), ("candidate", 163.4)):
    print("  %-9s %6.1f us -> %.2f TB/s effective" % (label, us, bytes_moved / (us * 1e-6) / 1e12))