#!/bin/bash
#SBATCH --job-name=e2e_plain_moe
#SBATCH --output=e2e_plain_moe_%j.out
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

SAVE_DIR="checkpoints/plain_moe_8e_k1"

# Plain MoE baseline: 8 full-size experts (intermediate=3072, matching
# the dense Qwen3-0.6B FFN), top-1 routing, Switch-style balance loss.

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type vanilla \
    --model Qwen/Qwen3-0.6B \
    --num-experts 8 \
    --top-k 1 \
    --expert-dim 3072 \
    --seq-len 8192 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --data-dir data/nemotron-moe-exam \
    --num-epochs 1 \
    --lr 2e-5 \
    --weight-decay 0.1 \
    --warmup-ratio 0.05 \
    --lr-scheduler cosine \
    --log-every 1 \
    --eval-every 500 \
    --seed 42 \
    --wandb-project moe-chunking-poc \
    --wandb-run-name plain-moe-qwen0.6b-8e-k1 \
    --save-dir "$SAVE_DIR" \
    --save-every 500 \
    --resume-from auto
