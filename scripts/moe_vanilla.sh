#!/bin/bash
#SBATCH --job-name=vanilla_moe
#SBATCH --output=temporal_moe_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:80gb:4
#SBATCH --partition=long
#SBATCH --time=3:00:00

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type vanilla \
    --model Qwen/Qwen3-0.6B \
    --num-experts 4 \
    --top-k 1 \
    --seq-len 256 \
    --batch-size 32 \
    --dataset-splits code math \
    --num-samples 1000000 \
    --num-steps 100000 \
    --lr 1e-4 \
    --log-every 10 \
    --eval-every 50 \
    --seed 42 \
    --wandb-project moe-chunking-poc
