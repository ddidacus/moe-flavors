#!/bin/bash
#SBATCH --job-name=e2e_temporal_noreg
#SBATCH --output=e2e_temporal_noreg_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:80gb:4
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1

SAVE_DIR="checkpoints/temporal_64e_k8_no_reg"

# ── Training ──

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type temporal \
    --model Qwen/Qwen3-0.6B \
    --num-experts 64 \
    --top-k 8 \
    --expert-dim 32 \
    --ratio-loss-alpha 0.0 \
    --entropy-alpha 0.0 \
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
    --wandb-run-name temporal-qwen0.6b-64e-k8-noreg \
    --save-dir "$SAVE_DIR" \
    --save-every 500 \
    --resume-from auto
