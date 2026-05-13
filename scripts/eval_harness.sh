#!/bin/bash
#SBATCH --job-name=moe_eval
#SBATCH --output=moe_eval_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:80gb:1
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00

cd /network/scratch/d/diego.calanzone/moe-playground
source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1

CHECKPOINT_DIR="${1:-checkpoints/step_100000}"

python scripts/eval_harness.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --tasks mmlu gsm8k \
    --batch-size auto \
    --output-dir "$CHECKPOINT_DIR/eval_results"
