#!/bin/bash
#SBATCH --job-name=sanity_humaneval
#SBATCH --output=sanity_humaneval_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --partition=long
#SBATCH --time=3:00:00

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache

python scripts/sanity_check_humaneval.py
