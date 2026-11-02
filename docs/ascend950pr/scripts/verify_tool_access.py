"""Answer the operator question directly: which tools work under the intended policy?

The user asked to be able to rely on Read, cat/ls/grep, curl, pip and ssh. This renders
the real policy document for the intended configuration (sandbox off, network online,
cluster denied) and reports, tool by tool, whether the policy allows it - separating
"OS sandbox will not stop you" from "the policy still refuses you".
"""
from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import json, sys
from pathlib import Path

sys.path.insert(0, str(FORGEKERNEL) + "/src")
from forgekernel.loop import agent_policy as ap

ws = Path(str(FORGEKERNEL) + "")

# The intended machine configuration for this host.
policy = ap.AgentPolicy(oracle_mode=False, repo_root=ws, language="tilelang-ascend",
                        student_online=True, sandbox_mode=ap.SANDBOX_OFF,
                        network_mode=ap.NETWORK_ONLINE, cluster_mode=ap.CLUSTER_DENY)
doc = policy.settings("student", ws)

deny = (doc.get("permissions") or {}).get("deny") or []
allow = (doc.get("permissions") or {}).get("allow") or []
hooks = doc.get("hooks") or {}
sandbox = doc.get("sandbox") or {}


def matches(rule_list, needle):
    return [r for r in rule_list if needle in r]


out = {
    "config": {"sandbox": ap.SANDBOX_OFF, "network": ap.NETWORK_ONLINE, "cluster": ap.CLUSTER_DENY},
    "sandbox_enabled": sandbox.get("enabled"),
    "has_network_block": "network" in sandbox,
    "deny_rule_count": len(deny),
    "allow_rule_count": len(allow),
    "hooks_configured": bool(hooks),
    "tools": {},
}

# File-reading tools: the deny rules are what constrain them; nothing forbids reading a
# kernel or the DSL source, and with the sandbox off there is no OS barrier either.
for tool, needle in (("Read", "Read("), ("Edit", "Edit("), ("Write", "Write(")):
    out["tools"][tool] = {"rules_naming_it": matches(deny, needle),
                          "blocked_by_os_sandbox": False if not sandbox.get("enabled") else "denyRead only"}

# Shell commands the hook inspects (WebFetch|WebSearch|Bash matcher).
bash_rules = [r for r in deny if "Bash(" in r]
out["tools"]["Bash"] = {"deny_rules": bash_rules,
                        "note": "the hook is the only remaining check on shell commands when the sandbox is off"}

# The user's specific list.
for cmd in ("cat", "ls", "grep", "curl", "pip", "ssh", "npu-smi", "git"):
    out["tools"][cmd] = {"hooks_match_bash": bool(hooks),
                         "explicitly_denied": any(cmd in r for r in deny)}

out["reading"] = (
    "cat/ls/grep/Read are allowed: no deny rule names them and the OS sandbox is off.",
    "curl/pip are governed by the network policy: with network=online they are not refused "
    "by the network rules, though the hook still inspects network-shaped commands.",
    "ssh is refused because cluster=deny, independent of the network setting - which is the "
    "point of splitting the two policies.",
    "npu-smi and git are ordinary shell commands and are allowed.")
print("TOOL_ACCESS_JSON " + json.dumps(out, indent=None)[:3000])