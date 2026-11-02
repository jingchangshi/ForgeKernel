#!/bin/bash
source /data/pri/cann-9.2.0/cann/set_env.sh >/dev/null 2>&1
export PATH=/home/s00653124/.conda/envs/s00653124_tilelang/bin:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_RT_VISIBLE_DEVICES=1 TILELANG_CACHE_DIR=/tmp/simsmoke/cache1
cd /tmp/simsmoke && rm -rf simout1
start=$(date +%s)
timeout 2400 msprof op simulator --soc-version=Ascend950PR_9579 --output=/tmp/simsmoke/simout1 --aic-metrics=PipeUtilization python3 /tmp/simsmoke/model1.py > /tmp/simsmoke/sim1.log 2>&1
rc=$?
end=$(date +%s)
echo "rc=$rc WALL_SECONDS=$((end-start))" >> /tmp/simsmoke/sim1.log
grep -vE "DRVSTUB" /tmp/simsmoke/sim1.log | tail -20
