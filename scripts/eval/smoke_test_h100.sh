#!/bin/bash
#SBATCH --job-name=smoke_h100
#SBATCH --output=/network/scratch/d/diego.calanzone/logs/smoke_h100_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=short-unkillable
#SBATCH --time=00:15:00

cd /home/mila/d/diego.calanzone/moe-flavors
source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache

python scripts/eval/eval_estimated_throughput.py --variant base \
    --num-expert-load-trials 16 --num-eval-prompts 8 --batch-size 4 \
    --out-dir /tmp/eval_estimated_throughput_h100_smoketest
