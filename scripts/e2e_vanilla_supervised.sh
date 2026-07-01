#!/bin/bash
#SBATCH --job-name=vanilla_sup
#SBATCH --output=vanilla_sup_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:80gb:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

SAVE_DIR="checkpoints/vanilla_moe_supervised"

accelerate launch --multi_gpu --num_processes=4 scripts/vanilla_moe_supervised.py \
    --model Qwen/Qwen3-0.6B \
    --num-experts 64 \
    --top-k 8 \
    --expert-dim 32 \
    --semantic-alpha 0.1 \
    --switch-lambda 0.01 \
    --tversky-alpha 0.3 \
    --tversky-beta 0.7 \
    --switch-tau 0.3 \
    --switch-temperature 0.1 \
    --seq-len 8192 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --dataset ddidacus/nemotron-moe-exam \
    --num-epochs 1 \
    --lr 4e-5 \
    --weight-decay 0.1 \
    --warmup-ratio 0.05 \
    --lr-scheduler cosine \
    --log-every 1 \
    --eval-every 10 \
    --seed 42 \
    --wandb-project moe-chunking-poc \
    --wandb-run-name vanilla-moe-supervised-qwen0.6b-64e-k8 \
    --save-dir "$SAVE_DIR" \
    --save-every 100 \
    --resume-from auto
