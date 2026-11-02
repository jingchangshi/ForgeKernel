#!/bin/bash
set -u
DEV=${DEV:-1}
SHAPE=${SHAPE:-8192x4096}
CAND=${CAND:-/home/s00653124/workspace/ForgeKernel/docs/ascend950pr/candidates/rmsnorm_simdvf_v5}
TAG=${TAG:-v5}
WORK=/home/s00653124/workspace/forgekernel-ascend950pr-work
source /data/pri/cann-9.2.0/cann/set_env.sh >/dev/null 2>&1
export PATH=/home/s00653124/.conda/envs/s00653124_tilelang/bin:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
idle=0
for attempt in $(seq 1 240); do
  busy=$(npu-smi info 2>/dev/null | awk -v d="$DEV" '/No running processes found in NPU/ { if ($NF+0==d) next } /^\| *[0-9]+ *\| *[0-9]+ *\|/ { print "p" }' | wc -l)
  procs=$(npu-smi info 2>/dev/null | sed -n "/Process id/,/^+/p" | awk -v d="$DEV" 'BEGIN{npu=-1} /^\| *[0-9]+ *\|/ && $3 ~ /^[0-9]+$/ {npud=$2} /^\| *[0-9]+ *\|.*python|pytest|bash/ {}' >/dev/null; echo 0)
  state=$(npu-smi info 2>/dev/null | tr -s " " | grep -E "^\| *[0-9]+ +[a-z]+ +[0-9]+" >/dev/null; echo ok)
  if npu-smi info 2>/dev/null | sed -n "/Process id/,/^+/p" | grep -qE "^\| *$DEV +[0-9]+ "; then idle=0; else idle=$((idle+1)); fi
  if [ "$idle" -ge 3 ]; then
    echo "IDLE_AT $(date -Is) device=$DEV attempt=$attempt"
    python3 /home/s00653124/workspace/AgentGates/agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py \
      --candidate "$CAND" --shape "$SHAPE" --device "$DEV" --pairs 15 --repeat 8 --child-timeout 600 \
      --cache "$WORK/accept-${TAG}-w$attempt" > "$WORK/accept-${TAG}-w$attempt.json" 2> "$WORK/accept-${TAG}-w$attempt.stderr"
    rc=$?
    echo "ATTEMPT $attempt rc=$rc at $(date -Is)"
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 1 ]; then exit 0; fi
    idle=0
  fi
  sleep 10
done
echo GAVE_UP
exit 2
