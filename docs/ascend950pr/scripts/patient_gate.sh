#!/bin/bash
set -u
W=/home/s00653124/workspace/forgekernel-ascend950pr-work
CAND=${CAND:-/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_v15}
TAG=${TAG:-v15}
DEV=${DEV:-0}
source /data/pri/cann-9.2.0/cann/set_env.sh >/dev/null 2>&1
export PATH=/home/s00653124/.conda/envs/s00653124_tilelang/bin:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
for try in $(seq 1 12); do
  python3 /home/s00653124/workspace/AgentGates/agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py --candidate "$CAND" --shape 8192x4096 --device "$DEV" --pairs 15 --repeat 8 --child-timeout 600 --cache "$W/pat-$TAG-$try" > "$W/pat-$TAG-$try.json" 2> "$W/pat-$TAG-$try.stderr"
  rc=$?
  q=$(python3 -c "
import json
try: d=json.load(open(\"$W/pat-$TAG-$try.json\"))
except Exception: print(-1); raise SystemExit
t=(d.get(\"timing\") or {}).get(\"8192x4096\") or {}
print(t.get(\"quietWindows\") if t.get(\"quietWindows\") is not None else -1)
" 2>/dev/null)
  echo "[pat] $TAG try=$try rc=$rc quiet=${q:-?} $(date -Is)"
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 1 ]; then echo "[pat] DECISIVE $try"; break; fi
  sleep 12
done
echo PATDONE
