from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import sys, json
from pathlib import Path
W = Path(str(WORK_AGENTGATES) + "")
sys.path.insert(0, str(W))
import importlib.util
spec = importlib.util.spec_from_file_location("acc", W / "agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py")
ACC = importlib.util.module_from_spec(spec); spec.loader.exec_module(ACC)

def wins(n_quiet, n_cont, util_quiet=0, util_cont=99, speed_quiet=1.05, speed_cont=0.30):
    out = []
    for _ in range(n_quiet):
        out.append({"vendor_ms": 0.01, "candidate_ms": 0.01/speed_quiet, "speedup": speed_quiet,
                    "witness": {"aicoreUtilBefore": util_quiet, "aicoreUtilAfter": util_quiet}})
    for _ in range(n_cont):
        out.append({"vendor_ms": 0.01, "candidate_ms": 0.01/speed_cont, "speedup": speed_cont,
                    "witness": {"aicoreUtilBefore": util_cont, "aicoreUtilAfter": util_cont}})
    return out

cases = [
  ("all contended (must RETEST, no quiet sample)", wins(0, 10)),
  ("all quiet, fast candidate (must MATCH/EXCEED)", wins(12, 0)),
  ("9 quiet + 3 contended = 75% (>=70 -> judged)", wins(9, 3)),
  ("6 quiet + 4 contended = 60% (<70 -> RETEST)", wins(6, 4)),
  ("8 quiet + 0 contended (<9 -> RETEST)", wins(8, 0)),
  ("2 usable + 8 unclassifiable (-> RETEST, unusable reason)", wins(2, 0) + [{"vendor_ms": 0.01, "candidate_ms": 0.01, "speedup": 1.0, "witness": {}}]*8),
]

print("%-52s %-9s %-6s %-8s %s" % ("case", "status", "quiet", "frac", "reason"))
for name, ws in cases:
    v = ACC.judge_timing(ws, "64x4096")
    print("%-52s %-9s %-6s %-8s %s" % (
        name, v["status"], v.get("decisionSampleWindows"),
        v.get("quietFractionOfWindows"), (v.get("why") or "")[:60]))

# Prove the contending windows are NOT in the decision sample: a candidate that is
# slow only while contended must not be rescued by them, and must not be condemned.
mixed = wins(12, 8, speed_quiet=1.05, speed_cont=0.20)
v = ACC.judge_timing(mixed, "8192x4096")
print()
print("12 quiet(1.05x) + 8 contended(0.20x) ->", v["status"],
      "point", round(v.get("pointSpeedup") or 0, 4),
      "decisionSampleWindows", v.get("decisionSampleWindows"))
print("  contended windows excluded:", v.get("decisionSampleWindows") == 12)
print("  point reflects quiet arms only:", abs((v.get("pointSpeedup") or 0) - 1.05) < 1e-9)