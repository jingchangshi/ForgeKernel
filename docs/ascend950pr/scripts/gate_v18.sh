#!/bin/bash
set -u
export PATH=/home/s00653124/.conda/envs/s00653124_tilelang/bin:$PATH
source /data/pri/cann-9.2.0/cann/set_env.sh >/dev/null 2>&1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
W=/home/s00653124/workspace/forgekernel-ascend950pr-work
for try in 1 2 3 4 5 6; do
  python3 /home/s00653124/workspace/AgentGates/agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py --candidate /home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_v18 --shape 8192x4096 --device 0 --pairs 15 --repeat 8 --child-timeout 600 --cache "$W/g18-$try" > "$W/g18-$try.json" 2> "$W/g18-$try.stderr"
  rc=$?
  q=$(python3 -c "
import json
try: d=json.load(open('$W/g18-$try.json'))
except Exception: print(-1); raise SystemExit
t=(d.get('timing') or {}).get('8192x4096') or {}
print(t.get('quietWindows') if t.get('quietWindows') is not None else -1)" 2>/dev/null)
  echo "[g18] try=$try rc=$rc quiet=$q $(date -Is)"
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 1 ]; then break; fi
  sleep 10
done
echo G18DONE
