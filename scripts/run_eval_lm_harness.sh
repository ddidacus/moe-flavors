#!/bin/bash
#SBATCH --job-name=eval_lm_harness
#SBATCH --output=eval_lm_harness_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00

# Takes one or more variant names as positional args, each pinned to its own
# GPU (CUDA_VISIBLE_DEVICES=0,1,...) and run in parallel as background
# processes within this single job -- same pattern as run_eval_soft_cache.sh's
# base/tuned split. Pass up to 3 variants (short-unkillable's QOS requires a
# minimum of 4 GPUs/job regardless of how many are actually used, hence 4
# requested above even for a 2-3 variant job). --out-dir defaults to
# evals/<today>; override with OUT_DIR=... if needed.
#
# Usage: sbatch scripts/run_eval_lm_harness.sh <variant1> [variant2] [variant3]
#        sbatch scripts/run_eval_lm_harness.sh merge   # CPU-only, no GPU needed
#
# Time budget: short-unkillable hard-caps at 3h, so NUM_SEEDS/LIMIT/
# MAX_GEN_TOKS are trimmed down from eval_lm_harness.py's own defaults
# (4 seeds/200/2048) to actually fit -- override via env vars if you have
# more time budget elsewhere (e.g. resubmit on `long` with the defaults).
NUM_SEEDS="${NUM_SEEDS:-2}"
LIMIT="${LIMIT:-50}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-1024}"
MATH_ONLY="${MATH_ONLY:-0}"   # 1: rerun+patch just hendrycks_math (see --math-only)

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache

OUT_DIR="${OUT_DIR:-evals/$(date +%F)}"

if [ "$1" = "merge" ]; then
    python scripts/eval_lm_harness.py --variant merge --out-dir "$OUT_DIR"
    exit $?
fi

MATH_ONLY_FLAG=""
if [ "$MATH_ONLY" = "1" ]; then MATH_ONLY_FLAG="--math-only"; fi

pids=()
gpu=0
for variant in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${variant} \
        python scripts/eval_lm_harness.py --variant "$variant" --out-dir "$OUT_DIR" \
        --num-seeds "$NUM_SEEDS" --limit "$LIMIT" --max-gen-toks "$MAX_GEN_TOKS" $MATH_ONLY_FLAG &
    pids+=($!)
    gpu=$((gpu + 1))
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit $status
