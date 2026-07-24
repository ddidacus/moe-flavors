#!/bin/bash
# Runs one or more scripts/eval_lm_harness.py variants in parallel
# (GPU-pinned), mirroring scripts/run_eval_lm_harness.sh's pattern. Not
# meant to be run directly -- invoked by scripts/cluv/eval_lm_harness.sh via
# `cluv submit <cluster> -- bash scripts/cluv/_eval_lm_harness_run.sh
# <cluster> <variant> [variant ...]`.
set -euo pipefail
CLUSTER="$1"; shift
OUT_DIR="evals/${CLUSTER}/$(date +%F)"

pids=()
gpu=0
for variant in "$@"; do
    ckpt_args=()
    if [ "$variant" != "base" ]; then
        ckpt_args=(--checkpoint-dir "checkpoints/${variant}_${CLUSTER}")
    fi
    CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${variant} \
        python scripts/eval_lm_harness.py --variant "$variant" "${ckpt_args[@]}" \
        --out-dir "$OUT_DIR" &
    pids+=($!)
    gpu=$((gpu + 1))
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
