from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import json,re,subprocess,os
raw=open(str(DSH_CREDENTIALS) + '').read()
m=re.search(r'MAGPIE_API_KEY\s*:\s*(\S+)', raw)
key=m.group(1).strip().strip('"').strip("'")
env=dict(os.environ, ANTHROPIC_BASE_URL='https://api.1031989.xyz', ANTHROPIC_API_KEY=key, ANTHROPIC_AUTH_TOKEN=key, ANTHROPIC_MODEL='claude-opus-5-5')
p=subprocess.run(['claude','-p','Reply with exactly: CANARY_OK','--model','claude-opus-5-5'],env=env,capture_output=True,text=True,timeout=180)
print('exit',p.returncode)
print('stdout',p.stdout.strip()[:300])
print('stderr',p.stderr.strip()[:300])