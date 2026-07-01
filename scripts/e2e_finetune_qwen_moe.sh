#!/bin/bash
#SBATCH --job-name=finetune_qwen_moe
#SBATCH --output=finetune_qwen_moe_%j.out
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

SAVE_DIR="checkpoints/finetune_qwen1.5_moe_a2.7b"

# 14.3B params total — needs DeepSpeed ZeRO-2 to fit on 4xA100 80GB
accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_zero2.json \
    --num_processes 4 \
    --mixed_precision bf16 \
    scripts/finetune_qwen_moe.py \
    --model Qwen/Qwen1.5-MoE-A2.7B \
    --dataset ddidacus/nemotron-moe-exam \
    --seq-len 4096 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --num-epochs 1 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.03 \
    --lr-scheduler cosine \
    --log-every 10 \
    --eval-every 500 \
    --seed 42 \
    --wandb-project moe-chunking-poc \
    --wandb-run-name finetune-qwen1.5-moe-a2.7b \
    --save-dir "$SAVE_DIR" \
    --save-every 500 \
    --resume-from auto
