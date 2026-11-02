"""Audit perf-report.md: every number must be traceable to an evidence file."""
from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import json, re
from pathlib import Path
P = Path(str(PLAN) + "")
rep = (P / "perf-report.md").read_text()
ev = json.load(open(P / "evidence/probe-bf16-2026-11-02.json"))
acc = json.load(open(P / "artifacts/accept-genuine.json"))

checks = []
probe = {r["shape"]: r for r in ev["bf16_rmsnorm_probe"]["results"]}
pre, dec = probe["8192x4096 bf16"], probe["64x4096 bf16"]
for label, val, text in (
    ("prefill start ratio", pre["speedup_vs_vendor"], "0.3148"),
    ("decode start ratio", dec["speedup_vs_vendor"], "0.8583"),
    ("prefill tilelang ms", pre["tilelang_ms"], "0.1634"),
    ("prefill vendor ms", pre["vendor_ms"], "0.0514")):
    checks.append((label, round(val, 4) == float(text), text in rep))

ma = ev["msprof_attribution_check"]["result"]
for k, v in (("aiv_vec", "84.9"), ("aiv_mte2", "32.0"), ("aiv_mte3", "12.6")):
    checks.append(("cand " + k, ma["tiled_pipes"][k] == v + "%", v + "%" in rep))
for k, v in (("aiv_vec", "80.7"), ("aiv_mte2", "64.2"), ("aiv_mte3", "44.9")):
    checks.append(("vendor " + k, ma["vendor_pipes"][k] == v + "%", v + "%" in rep))

rs = ev["row_tile_sweep"]["results"]
checks.append(("rows=1 289us", round(rs["rows=1"]["ms"] * 1000) == 289, "289" in rep))
checks.append(("rows=4 164us", round(rs["rows=4"]["ms"] * 1000) == 164, "164" in rep))
checks.append(("rows=2 190us", round(rs["rows=2"]["ms"] * 1000) == 190, "189.6" in rep))
checks.append(("rows=8 UB 271360", "271360" in rs["rows=8"]["error"], "271360" in rep))

t = acc["timing"]
for shape, label in (("8192x4096", "8192×4096"), ("64x4096", "64×4096")):
    pt, (lo, hi) = t[shape]["pointSpeedup"], t[shape]["interval95"]
    row = [ln for ln in rep.splitlines() if ln.startswith("| " + label)]
    found = bool(row)
    ptx = ("%.4f" % pt) in rep
    lox = ("%.4f" % lo) in rep
    hix = ("%.4f" % hi) in rep
    checks.append((label + " point %.4f" % pt, found and ptx, ptx))
    checks.append((label + " CI low %.4f" % lo, lox, lox))
    checks.append((label + " CI high %.4f" % hi, hix, hix))

cb = ev["contention_bias_measured_2026-11-02"]["measurements"]
checks.append(("bias 8192 0.3224", round(cb["8192x4096"]["idle"]["ratio"], 4) == 0.3224, "0.3224" in rep))
checks.append(("bias 8192 0.6615", round(cb["8192x4096"]["contended"]["ratio"], 4) == 0.6615, "0.6615" in rep))
checks.append(("bias 64 0.8273", round(cb["64x4096"]["contended"]["ratio"], 4) == 0.8273, "0.8273" in rep))

cd = ev["contention_bias_measured_2026-11-02"]["canary_detector_quality"]
checks.append(("canary 3/9", cd["detectionsAbove1_15"] == 3 and cd["windows"] == 9, "3/9" in rep))

print("%-34s %-11s %-9s" % ("claim", "in-evidence", "in-report"))
bad = 0
for name, in_ev, in_rep in checks:
    ok = in_ev and in_rep
    bad += 0 if ok else 1
    print("%-34s %-11s %-9s%s" % (name, in_ev, in_rep, "" if ok else "  <-- MISMATCH"))
print()
print("checked", len(checks), "claims;", bad, "problem(s)")