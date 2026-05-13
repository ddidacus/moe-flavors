#!/bin/bash
#SBATCH --job-name=temporal_moe
#SBATCH --output=temporal_moe_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:80gb:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00

cd /network/scratch/d/diego.calanzone/moe-playground
source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type temporal \
    --model Qwen/Qwen3-0.6B \
    --num-experts 4 \
    --top-k 2 \
    --ratio-loss-N 3 \
    --ratio-loss-alpha 0.03 \
    --entropy-threshold 0.1 \
    --entropy-alpha 0.05 \
    --entropy-warmup-steps 500 \
    --seq-len 256 \
    --batch-size 32 \
    --dataset-splits code math \
    --num-samples 1000000 \
    --num-steps 100000 \
    --lr 1e-4 \
    --log-every 10 \
    --eval-every 50 \
    --seed 42 \
    --wandb-project moe-chunking-poc \
    --save-dir /network/scratch/d/diego.calanzone/moe-playground/checkpoints \
    --save-every 10000
