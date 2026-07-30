#!/bin/bash
# Runs one or more scripts/eval_soft_cache.py variants in parallel
# (GPU-pinned). Not meant to be run directly -- invoked by
# scripts/cluv/eval_soft_cache.sh via `cluv submit <cluster> -- bash
# scripts/cluv/_eval_soft_cache_run.sh <cluster> <variant> [variant ...]`.
set -euo pipefail
CLUSTER="$1"; shift
OUT_DIR="evals/${CLUSTER}/soft_cache_$(date +%F)"

pids=()
gpu=0
for variant in "$@"; do
    ckpt_args=()
    if [ "$variant" != "base" ]; then
        ckpt_args=(--checkpoint-dir "checkpoints/${variant}_${CLUSTER}")
    fi
    CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${variant} \
        python scripts/eval_soft_cache.py --variant "$variant" "${ckpt_args[@]}" \
        --out-dir "$OUT_DIR" &
    pids+=($!)
    gpu=$((gpu + 1))
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
