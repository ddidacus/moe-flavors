#!/bin/bash
#SBATCH --job-name=eval_lm_harness
#SBATCH --output=eval_lm_harness_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:l40s:2
#SBATCH --partition=long
#SBATCH --time=16:00:00

# Takes one or more variant names as positional args, each pinned to its own
# GPU (CUDA_VISIBLE_DEVICES=0,1,...) and run in parallel as background
# processes within this single job -- same pattern as run_eval_soft_cache.sh's
# base/tuned split. Pass 1 variant for a solo run, 2 to fan out in parallel
# (up to the 2 GPUs requested above -- main's QOS caps gres/gpu at 2/user
# total, hence `long` + 2 GPUs here). --out-dir defaults to evals/<today>;
# override with OUT_DIR=... if needed.
#
# Usage: sbatch scripts/run_eval_lm_harness.sh <variant1> [variant2] ...
#        sbatch scripts/run_eval_lm_harness.sh merge   # CPU-only, no GPU needed
#
# Time budget: gsm8k/humaneval/hendrycks_math (generate_until, up to 2048
# new tokens each) run once per seed in SEEDS (4x by default, see
# eval_lm_harness.py) -- this is the dominant cost.

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache

OUT_DIR="${OUT_DIR:-evals/$(date +%F)}"

if [ "$1" = "merge" ]; then
    python scripts/eval_lm_harness.py --variant merge --out-dir "$OUT_DIR"
    exit $?
fi

pids=()
gpu=0
for variant in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${variant} \
        python scripts/eval_lm_harness.py --variant "$variant" --out-dir "$OUT_DIR" &
    pids+=($!)
    gpu=$((gpu + 1))
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit $status
