#!/bin/bash
#SBATCH --job-name=smoke_eval_complete_olmoe
#SBATCH --output=/network/scratch/d/diego.calanzone/logs/smoke_eval_complete_olmoe_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=short-unkillable
#SBATCH --time=00:30:00

cd /home/mila/d/diego.calanzone/moe-flavors
source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export HF_ALLOW_CODE_EVAL=1

python scripts/eval/eval_complete.py --model allenai/OLMoE-1B-7B-0125-Instruct \
    --variant base --cache-size 16 --cache-experts-per-token 8 --cache-topk \
    --num-eval-prompts 8 --num-viz-prompts 4 \
    --num-expert-load-trials 16 --skip-parts 1 \
    --gen-len 32 --batch-size 4 \
    --out-dir-root /tmp/eval_complete_olmoe_smoketest
