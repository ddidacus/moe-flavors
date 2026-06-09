#!/bin/bash
#SBATCH --job-name=e2e_temporal
#SBATCH --output=e2e_temporal_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=long
#SBATCH --time=12:00:00

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1
# export PYTHONPATH="/network/scratch/d/diego.calanzone/moe-playground/stubs:${PYTHONPATH:-}"

SAVE_DIR="checkpoints/temporal_8e_k2_8192"

# ── Training ──

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type temporal \
    --model Qwen/Qwen3-0.6B \
    --num-experts 8 \
    --top-k 2 \
    --ratio-loss-N 3 3 4 4 5 5 6 6 7 7 7 8 8 9 9 10 10 11 11 11 12 12 13 13 14 14 15 15 \
    --ratio-loss-alpha 0.03 \
    --entropy-threshold 0.1 \
    --entropy-alpha 0.05 \
    --entropy-warmup-steps 500 \
    --seq-len 8192 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --data-dir data/nemotron-moe-exam \
    --num-epochs 1 \
    --lr 8e-4 \
    --log-every 10 \
    --eval-every 500 \
    --seed 42 \
    --wandb-project moe-chunking-poc \
    --wandb-run-name temporal-moe-N-9 \
    --save-dir "$SAVE_DIR" \
    --save-every 500 \
    --resume-from auto
