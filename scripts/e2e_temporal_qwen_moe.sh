#!/bin/bash
#SBATCH --job-name=temporal_qwen_moe
#SBATCH --output=temporal_qwen_moe_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:80gb:4
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@120

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1

SAVE_DIR="checkpoints/temporal_qwen1.5_moe_64e_k8"

# Loads Qwen1.5-MoE-A2.7B (14.3B), replaces all MoE layers with temporal
# chunking MoE (64 experts, expert_dim=64, top-k=8). After conversion the
# model is ~1.6B params, so plain DDP is fine.
# ratio_loss_N: 24 values for 24 layers, linearly increasing segment count.

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type temporal \
    --model Qwen/Qwen1.5-MoE-A2.7B \
    --num-experts 64 \
    --top-k 8 \
    --expert-dim 64 \
    --ratio-loss-N 3 3 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 \
    --ratio-loss-alpha 0.03 \
    --entropy-threshold 0.1 \
    --entropy-alpha 0.05 \
    --entropy-warmup-steps 500 \
    --seq-len 4096 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --dataset ddidacus/nemotron-moe-exam \
    --num-epochs 1 \
    --lr 4e-5 \
    --weight-decay 0.1 \
    --warmup-ratio 0.05 \
    --lr-scheduler cosine \
    --log-every 1 \
    --eval-every 500 \
    --seed 42 \
    --wandb-project moe-chunking-poc \
    --wandb-run-name temporal-qwen1.5-moe-64e-k8 \
    --save-dir "$SAVE_DIR" \
    --save-every 500 \
    --resume-from auto
