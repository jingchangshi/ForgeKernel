#!/usr/bin/env bash
set -euo pipefail
python3 kernel_run.py > kernel_out.txt 2>&1 || { cat kernel_out.txt; exit 1; }
cat kernel_out.txt
echo "[gpu_entry.sh] task=$1 finished"
