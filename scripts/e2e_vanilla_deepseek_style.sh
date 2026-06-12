#!/bin/bash
#SBATCH --job-name=e2e_vanilla_ds
#SBATCH --output=e2e_vanilla_ds_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --constraint=80gb
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1

SAVE_DIR="checkpoints/vanilla_64e_k8_2048_deepseek_style"

# ── Training ──

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type vanilla \
    --model Qwen/Qwen3-0.6B \
    --num-experts 64 \
    --top-k 8 \
    --expert-dim 32 \
    --seq-len 8192 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --data-dir data/nemotron-moe-exam \
    --num-epochs 1 \
    --lr 4e-5 \
    --weight-decay 0.1 \
    --warmup-ratio 0.05 \
    --lr-scheduler cosine \
    --log-every 1 \
    --eval-every 500 \
    --seed 42 \
    --wandb-project moe-chunking-poc \
    --wandb-run-name vanilla-qwen1.8b-64e-k8 \
    --save-dir "$SAVE_DIR" \
    --save-every 500 \
    --resume-from auto
