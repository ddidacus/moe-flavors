#!/bin/bash
#SBATCH --job-name=e2e_vanilla
#SBATCH --output=e2e_vanilla_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:80gb:4
#SBATCH --partition=long
#SBATCH --time=6:00:00

set -euo pipefail

cd /network/scratch/d/diego.calanzone/moe-playground
source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/network/scratch/d/diego.calanzone/moe-playground/stubs:${PYTHONPATH:-}"

SAVE_DIR="checkpoints/vanilla_${SLURM_JOB_ID:-local}"

# ── Training ──

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type vanilla \
    --model Qwen/Qwen3-0.6B \
    --num-experts 4 \
    --top-k 2 \
    --seq-len 256 \
    --batch-size 32 \
    --dataset-splits code math \
    --num-samples 1000000 \
    --num-steps 10000 \
    --lr 1e-4 \
    --log-every 10 \
    --eval-every 50 \
    --seed 42 \
    --wandb-project moe-chunking-poc \
    --save-dir "$SAVE_DIR" \
    --save-every 10000

# ── Evaluation ──

FINAL_CKPT=$(ls -d "$SAVE_DIR"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)

if [ -z "$FINAL_CKPT" ]; then
    echo "ERROR: No checkpoint found in $SAVE_DIR"
    exit 1
fi

echo "Evaluating checkpoint: $FINAL_CKPT"

python scripts/eval_harness.py \
    --checkpoint-dir "$FINAL_CKPT" \
    --tasks mmlu gsm8k \
    --batch-size auto \
    --output-dir "$FINAL_CKPT/eval_results"
