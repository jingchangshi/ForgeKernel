from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import json,os,subprocess,sys,re
raw=open(str(DSH_CREDENTIALS) + "").read()
m=re.search(r"MAGPIE_API_KEY\s*:\s*(\S+)",raw)
key=m.group(1).strip().strip(chr(34)).strip(chr(39))
model=sys.argv[1]
env=dict(os.environ,ANTHROPIC_BASE_URL="https://api.1031989.xyz",ANTHROPIC_API_KEY=key,ANTHROPIC_AUTH_TOKEN=key)
p=subprocess.run(["claude","-p","Reply with exactly: CANARY_OK","--model",model],env=env,capture_output=True,text=True,timeout=240)
ok="CANARY_OK" in p.stdout
print(json.dumps({"model":model,"exit":p.returncode,"ok":ok,"stdout":p.stdout.strip()[:160],"stderr":p.stderr.strip()[:200]}))