#!/bin/bash
set -u
export PATH=/home/s00653124/.conda/envs/s00653124_tilelang/bin:$PATH
source /data/pri/cann-9.2.0/cann/set_env.sh >/dev/null 2>&1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
W=/home/s00653124/workspace/forgekernel-ascend950pr-work
C=/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates
run_one() {
  local tag="$1" cand="$2"
  for try in 1 2 3 4 5 6; do
    python3 /home/s00653124/workspace/AgentGates/agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py --candidate "$cand" --shape 8192x4096 --device 0 --pairs 15 --repeat 8 --child-timeout 600 --cache "$W/$tag-$try" > "$W/$tag-$try.json" 2> "$W/$tag-$try.stderr"
    rc=$?
    q=$(python3 -c "
import json
try: d=json.load(open('$W/$tag-$try.json'))
except Exception: print(-1); raise SystemExit
t=(d.get('timing') or {}).get('8192x4096') or {}
print(t.get('quietWindows') if t.get('quietWindows') is not None else -1)" 2>/dev/null)
    echo "[$tag] try=$try rc=$rc quiet=$q $(date -Is)"
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 1 ]; then return 0; fi
    sleep 10
  done
  return 1
}
run_one v26p "$C/rmsnorm_simdvf_v26"
run_one v27p "$C/rmsnorm_simdvf_v27"
echo V2627DONE
