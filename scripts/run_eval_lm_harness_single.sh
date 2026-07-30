#!/bin/bash
#SBATCH --job-name=eval_lm_harness_1gpu
#SBATCH --output=eval_lm_harness_1gpu_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --time=10:00:00

# Single-variant, single-GPU version of run_eval_lm_harness.sh -- avoids
# short-unkillable's 4-GPU-per-job QOS minimum (and its GRES contention with
# other 4-GPU training jobs) when only one variant needs (re-)evaluating.
# Usage: sbatch scripts/run_eval_lm_harness_single.sh <variant>

NUM_SEEDS="${NUM_SEEDS:-2}"
LIMIT="${LIMIT:-50}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-1024}"

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache

OUT_DIR="${OUT_DIR:-evals/$(date +%F)}"

python scripts/eval_lm_harness.py --variant "$1" --out-dir "$OUT_DIR" \
    --num-seeds "$NUM_SEEDS" --limit "$LIMIT" --max-gen-toks "$MAX_GEN_TOKS"
