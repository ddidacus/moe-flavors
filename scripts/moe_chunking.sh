#!/bin/bash
#SBATCH --job-name=temporal_moe
#SBATCH --output=temporal_moe_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache

accelerate launch --multi_gpu --num_processes=4 moe_mixin_poc.py
# python moe_mixin_poc.py