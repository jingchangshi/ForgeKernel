#!/bin/bash
set -u
W=/home/s00653124/workspace/forgekernel-ascend950pr-work
C=/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates
source /data/pri/cann-9.2.0/cann/set_env.sh >/dev/null 2>&1
export PATH=/home/s00653124/.conda/envs/s00653124_tilelang/bin:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# wait for the first sweep to finish before competing for windows
for i in $(seq 1 120); do
  if grep -q ALLDONE "$W/sweep_variants.log" 2>/dev/null; then break; fi
  sleep 20
done
run_gate() {
  local tag="$1" shape="$2" cand="$3"
  for try in 1 2 3 4 5 6 7 8; do
    if npu-smi info 2>/dev/null | sed -n "/Process id/,/^+/p" | grep -qE "^\| *1 +[0-9]+ "; then sleep 25; continue; fi
    python3 /home/s00653124/workspace/AgentGates/agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py --candidate "$cand" --shape "$shape" --device 1 --pairs 15 --repeat 8 --child-timeout 600 --cache "$W/s2-$tag-$try" > "$W/s2-$tag-$try.json" 2> "$W/s2-$tag-$try.stderr"
    rc=$?
    echo "[s2] $tag try=$try rc=$rc $(date -Is)"
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 1 ]; then return 0; fi
  done
  return 1
}
run_gate v10 8192x4096 $C/rmsnorm_simdvf_v10
run_gate v9 8192x4096 $C/rmsnorm_simdvf_v9
run_gate dec_v5 64x4096 $C/rmsnorm_simdvf_v5
echo S2DONE
