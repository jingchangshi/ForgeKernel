#!/bin/bash
set -u
W=/home/s00653124/workspace/forgekernel-ascend950pr-work
C=/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates
source /data/pri/cann-9.2.0/cann/set_env.sh >/dev/null 2>&1
export PATH=/home/s00653124/.conda/envs/s00653124_tilelang/bin:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
shapes="8192x4096"
for v in v6 v7 v8 v9; do
  ok=0
  for try in 1 2 3 4 5 6; do
    idle=0
    for p in 1 2 3; do
      if npu-smi info 2>/dev/null | sed -n "/Process id/,/^+/p" | grep -qE "^\| *1 +[0-9]+ "; then idle=0; else idle=$((idle+1)); fi
      sleep 5
    done
    [ "$idle" -lt 3 ] && continue
    python3 /home/s00653124/workspace/AgentGates/agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py --candidate "$C/rmsnorm_simdvf_$v" --shape 8192x4096 --device 1 --pairs 15 --repeat 8 --child-timeout 600 --cache "$W/acc-$v-t$try" > "$W/acc-$v-t$try.json" 2> "$W/acc-$v-t$try.stderr"
    rc=$?
    echo "$v try=$try rc=$rc $(date -Is)"
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 1 ]; then ok=1; break; fi
  done
  echo "$v done ok=$ok"
done
echo ALLDONE
