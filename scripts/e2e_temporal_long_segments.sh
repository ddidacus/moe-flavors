#!/bin/bash
#SBATCH --job-name=e2e_temporal_longseg
#SBATCH --output=e2e_temporal_longseg_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:80gb:4
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120

source .venv/bin/activate
#export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
#export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
#export PYTHONDONTWRITEBYTECODE=1

mkdir -p checkpoints
SAVE_DIR="checkpoints/temporal_64e_k8_long_segments"

# ── Training ──
# Target segment lengths linearly spaced from 64 to 4096 across 28 layers
# N = seq_len / seg_len (rounded), no regularizers (LM loss only)

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type temporal \
    --model Qwen/Qwen3-0.6B \
    --num-experts 64 \
    --top-k 8 \
    --expert-dim 32 \
    --ratio-loss-N 128 50 31 23 18 15 12 11 10 9 8 7 7 6 6 5 5 5 4 4 4 4 4 4 3 3 3 3 \
    --ratio-loss-alpha 0.03 \
    --entropy-threshold 0.1 \
    --entropy-alpha 0.05 \
    --entropy-warmup-steps 500 \
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
    --wandb-run-name temporal-qwen0.6b-64e-k8-longseg \
    --save-dir "$SAVE_DIR" \
    --save-every 500 \
    --resume-from auto
