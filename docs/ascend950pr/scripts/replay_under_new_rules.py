from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import sys, json, importlib.util
from pathlib import Path
W = Path(str(WORK_AGENTGATES) + "")
P = Path(str(PLAN) + "")

spec = importlib.util.spec_from_file_location("acc", W / "agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py")
ACC = importlib.util.module_from_spec(spec); spec.loader.exec_module(ACC)

print("Replaying the recorded windows of the clean 12:53 run through the NEW rules.")
print("The question: does excluding contended/sparse windows change the published verdict?")
print()
for f, label in ((P/"artifacts/accept-genuine.json", "clean 12:53 run"),
                 (P/"artifacts/contended-run.json", "contended 13:03 run")):
    data = json.load(open(f))
    print("==", label, "==")
    for shape, sh in (data.get("timing") or {}).items():
        windows = sh.get("windows") or []
        verdict = ACC.judge_timing(windows, shape)
        old = sh.get("status")
        new = verdict["status"]
        print("  %-10s old=%-7s new=%-7s  decided=%s quiet=%s contended=%s void=%s" % (
            shape, old, new, verdict.get("decisionSampleWindows"),
            verdict.get("quietWindows"), verdict.get("contendedWindows"),
            verdict.get("voidWindows")))
        if verdict.get("pointSpeedup") is not None:
            print("               point=%.4f  interval=%s" % (
                verdict["pointSpeedup"], [round(x, 4) for x in verdict.get("interval95") or []]))
        print("               changed：" , "NO" if old == new else "YES")
    print()