#!/bin/bash
# Runs two full shell commands in parallel within one job, each pinned to
# its own GPU (CUDA_VISIBLE_DEVICES=0 / =1) -- the "2 eval scripts per job,
# 1 GPU + 4 CPUs each" packing scheme. If only one command is given, runs
# it alone on GPU 0.
#
# Usage: bash scripts/eval/run_pair.sh "<full command 1>" ["<full command 2>"]
set -uo pipefail
CMD1="$1"
CMD2="${2:-}"

status=0
CUDA_VISIBLE_DEVICES=0 bash -c "$CMD1" &
pid1=$!
pid2=""
if [ -n "$CMD2" ]; then
    CUDA_VISIBLE_DEVICES=1 bash -c "$CMD2" &
    pid2=$!
fi
wait "$pid1" || status=1
if [ -n "$pid2" ]; then
    wait "$pid2" || status=1
fi
exit $status
