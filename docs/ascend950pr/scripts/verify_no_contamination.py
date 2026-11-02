from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import sys, importlib.util, json
from pathlib import Path
W = Path(str(WORK_AGENTGATES) + "")
sys.path.insert(0, str(W))
spec = importlib.util.spec_from_file_location("acc", W / "agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py")
ACC = importlib.util.module_from_spec(spec); spec.loader.exec_module(ACC)

def wins(n_quiet, n_cont, sq, sc):
    out = []
    for _ in range(n_quiet):
        out.append({"vendor_ms": 0.01, "candidate_ms": 0.01/sq, "speedup": sq,
                    "witness": {"aicoreUtilBefore": 0, "aicoreUtilAfter": 0}})
    for _ in range(n_cont):
        out.append({"vendor_ms": 0.01, "candidate_ms": 0.01/sc, "speedup": sc,
                    "witness": {"aicoreUtilBefore": 99, "aicoreUtilAfter": 99}})
    return out

# The decisive test: choose a mix that PASSES the quiet-fraction gate (>=70%) so a
# verdict really is produced, then check the contaminated windows did not leak in.
ws = wins(15, 5, sq=1.02, sc=0.20)   # 75% quiet -> judged
v = ACC.judge_timing(ws, "8192x4096")
print("verdict:", v["status"])
print("decisionSampleWindows:", v.get("decisionSampleWindows"), "(must be 15, not 20)")
print("pointSpeedup:", v.get("pointSpeedup"), "(must be 1.02, the quiet arm)")
print("contendedWindows reported:", v.get("contendedWindows"), "/", v.get("quietWindows"))
print("interval:", v.get("interval95"))
print()
leak = abs((v.get("pointSpeedup") or 0) - 1.02) > 1e-9
print("VERDICT INFO LEAKED FROM CONTENDED WINDOWS:", leak)
print()
# And the wash-to-PASS scenario the contraction enables if windows were NOT excluded:
print("If contended windows were allowed to contaminate 0.478 truth, the 2.05x inflation")
print("would report it as %.3f, which clears the MATCH lower bound 0.98." % (0.478*2.052))