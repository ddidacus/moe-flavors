#!/bin/bash
#SBATCH --job-name=e2e_deepseek
#SBATCH --output=e2e_deepseek_%j.out
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

SAVE_DIR="checkpoints/deepseek_64e_4s_k4_deepseek_style"

# ── Training ──

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type deepseek \
    --model Qwen/Qwen3-0.6B \
    --num-experts 64 \
    --num-shared-experts 4 \
    --top-k 4 \
    --expert-dim 32 \
    --aux-loss-coeff 0.01 \
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
    --wandb-run-name deepseek-qwen0.6b-64e-4s-k4 \
    --save-dir "$SAVE_DIR" \
    --save-every 500 \
    --resume-from auto
