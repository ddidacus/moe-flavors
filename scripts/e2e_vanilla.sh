#!/bin/bash
#SBATCH --job-name=e2e_vanilla
#SBATCH --output=e2e_vanilla_%j.out
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

SAVE_DIR="checkpoints/vanilla_8e_k2_8192"

# ── Training ──

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type vanilla \
    --model Qwen/Qwen3-0.6B \
    --num-experts 8 \
    --top-k 2 \
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
    --wandb-run-name vanilla-moe \
    --save-dir "$SAVE_DIR" \
    --save-every 500 \
    --resume-from auto
