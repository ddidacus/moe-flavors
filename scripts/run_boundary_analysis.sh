#!/bin/bash
#SBATCH --job-name=boundary_analysis
#SBATCH --output=boundary_analysis_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --partition=long
#SBATCH --time=01:00:00

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1

python scripts/analyze_boundary_alignment.py \
    --checkpoint-dir checkpoints/temporal_64e_k8_long_segments/step_1000 \
    --dataset-name ddidacus/nemotron-moe-exam \
    --max-len 8192 \
    --batch-size 1 \
    --num-samples 100 \
    --num-viz-samples 10 \
    --tolerance 5 \
    --device cuda
