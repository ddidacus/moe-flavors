#!/bin/bash
# Runs two shell scripts in parallel within one job, each pinned to its own
# GPU (CUDA_VISIBLE_DEVICES=0 / =1) -- the "2 eval scripts per job, 1 GPU +
# 4 CPUs each" packing scheme. Takes FILE PATHS (not inline command
# strings) -- long multi-flag command strings get mangled somewhere in
# cluv's own argument-forwarding layer between the local `cluv submit ...
# -- program args` call and the remote sbatch invocation (reproduced: a
# long string arrived as a no-op, a short one like "echo hi" survived
# intact) -- writing each command to its own script file and passing a
# plain path sidesteps that entirely. If only one path is given, runs it
# alone on GPU 0.
#
# Usage: bash scripts/eval/run_pair.sh <script1.sh> [<script2.sh>]
set -uo pipefail
SCRIPT1="$1"
SCRIPT2="${2:-}"

status=0
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT1" &
pid1=$!
pid2=""
if [ -n "$SCRIPT2" ]; then
    CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT2" &
    pid2=$!
fi
wait "$pid1" || status=1
if [ -n "$pid2" ]; then
    wait "$pid2" || status=1
fi
exit $status
