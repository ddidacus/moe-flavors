#!/bin/bash
#SBATCH --job-name=smoke_eval_complete_cc
#SBATCH --output=/network/scratch/d/diego.calanzone/logs/smoke_eval_complete_cc_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=short-unkillable
#SBATCH --time=00:30:00

cd /home/mila/d/diego.calanzone/moe-flavors
source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export HF_ALLOW_CODE_EVAL=1

python scripts/eval/eval_complete_cache_conditioned.py \
    --model microsoft/Phi-tiny-MoE-instruct \
    --checkpoint-dir checkpoints/prompt_conditioned_tamia_200steps \
    --cache-sizes 2,4 --cache-experts-per-token 2 --cache-topk \
    --num-eval-prompts 8 --num-viz-prompts 4 \
    --num-expert-load-trials 16 --harness-total-budget 8 --harness-num-seeds 1 \
    --gen-len 32 --batch-size 4 \
    --out-dir-root /tmp/eval_complete_cc_smoketest
