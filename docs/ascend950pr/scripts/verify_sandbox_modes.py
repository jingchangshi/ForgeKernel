"""Lead-side check: sandbox off must disable ONLY the OS sandbox.

The risk in P1 is that turning the sandbox off quietly also removes the deny rules, the
PreToolUse hooks, or the judge/resume policy paths - i.e. the run loses protections nobody
intended to drop. This renders the policy document under all three modes and diffs them,
so the difference is visible rather than asserted.
"""
from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import json, sys
from pathlib import Path

sys.path.insert(0, str(FORGEKERNEL) + "/src")
from forgekernel.loop import agent_policy as ap

ws = Path(str(FORGEKERNEL) + "")
out = {"modes": {}}


def build(mode):
    policy = ap.AgentPolicy(oracle_mode=False, repo_root=ws, language="tilelang-ascend",
                            student_online=True, sandbox_mode=mode)
    return policy.settings("student", ws)


docs = {}
for mode in ap.SANDBOX_MODES:
    try:
        docs[mode] = build(mode)
        out["modes"][mode] = {"ok": True}
    except Exception as exc:
        out["modes"][mode] = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:200]}

if len(docs) == 3:
    strict, auto, off = docs["strict"], docs["auto"], docs["off"]
    out["sandbox_block"] = {
        "strict": strict.get("sandbox"),
        "auto": auto.get("sandbox"),
        "off": off.get("sandbox"),
    }
    # Protections that must survive every mode.
    for key in ("permissions", "hooks"):
        out.setdefault("invariance", {})[key] = {
            "strict_equals_off": strict.get(key) == off.get(key),
            "auto_equals_off": auto.get(key) == off.get(key),
        }
    deny_strict = len((strict.get("permissions") or {}).get("deny") or [])
    deny_off = len((off.get("permissions") or {}).get("deny") or [])
    out["deny_rule_count"] = {"strict": deny_strict, "off": deny_off,
                              "same": deny_strict == deny_off}
    out["hooks_present_off"] = bool(off.get("hooks"))
    out["only_sandbox_differs"] = (
        {k: v for k, v in strict.items() if k != "sandbox"}
        == {k: v for k, v in off.items() if k != "sandbox"})

print("POLICY_MODES_JSON " + json.dumps(out, indent=None)[:4000])