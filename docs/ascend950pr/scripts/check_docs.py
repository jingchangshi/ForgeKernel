from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import json,re
from pathlib import Path
root=Path(str(PLAN) + "")
fence=chr(96)*3
for p in sorted(root.glob("*.md")):
    t=p.read_text()
    assert t.count(fence)%2==0, p
    for target in re.findall(r"\]\(([^)]+)\)",t):
        if not target.startswith(("http:","https:")):
            assert (p.parent/target).exists(),(p,target)
    print("%-16s %4d lines, links ok"%(p.name,len(t.splitlines())))
json.load(open(root/"evidence/probe-bf16-2026-11-02.json"))
json.load(open(root/"evidence/environment-2026-11-02.json"))
print("evidence JSON parses")