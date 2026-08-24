#!/bin/bash
#SBATCH --job-name=cache_local_compare
#SBATCH --output=cache_local_compare_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

# Runs cache_sft and cache_reward side by side, in one job, on this mila
# A100L node: 2 GPUs each (CUDA_VISIBLE_DEVICES 0,1 vs 2,3, accelerate
# --num_processes 2 each), same wandb project so their curves (reward,
# eval/ppl, and the new eval/cache_hit_rate metric -- see
# CacheHitRateEvalCallback in finetune_moe_grpo.py, logged every
# --eval-hitrate-every steps) can be compared directly.
#
# Config matches scripts/cluv/train_cache_sft.sh / train_cache_reward.sh
# (1024+1024 context, batch=16, no grad accumulation, 200 steps, 3200
# samples -- see that script's comment for the smoke-test rationale). A
# full 200-step run takes ~7.8h at the per-step rate measured on a single
# A100L (scripts/train/smoke_test_context.sh), well past short-unkillable's
# 3h cap -- this job will get killed mid-run; --save-every 50 plus
# --signal=B:USR1@120 (PreemptionCallback checkpoints cleanly 2 minutes
# before the limit) means whatever progress was made is safely checkpointed
# and --resume will pick it back up on a resubmit. This is meant as a
# "watch the curves for a while" diagnostic run, not a guaranteed full
# completion in one job.
#
# Usage: sbatch scripts/train/run_cache_local_compare.sh

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(--dataset nvidia/Nemotron-Post-Training-Dataset-v2
             --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr
             --max-samples 3200 --prompt-len 1024 --completion-len 1024
             --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42
             --wandb-project moe-cache-reinforce
             --save-every 50 --save-total-limit 3 --resume
             --batch-size 16 --gradient-accumulation-steps 1 --num-steps 200
             --num-generations 8 --temperature 1.0 --rl-coef 2.0 --beta 0.08
             --cache-size 4 --cache-layer -1 --cache-experts-per-token 2
             --cache-topk --soft-cache
             --eval-ppl-every 10 --eval-hitrate-every 25)

# Two independent `accelerate launch --multi_gpu` process groups on one
# node need distinct rendezvous ports, or their default (29500) collides.
TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}_cache_sft" \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
    --main_process_port 29500 \
    scripts/train/finetune_moe_grpo.py "${COMMON_ARGS[@]}" \
    --sft-coef 0.5 \
    --wandb-run-name "cache-sft-local-${SLURM_JOB_ID}" \
    --save-dir "checkpoints/cache_sft_local_${SLURM_JOB_ID}" \
    > "cache_sft_local_${SLURM_JOB_ID}.log" 2>&1 &
pid_sft=$!

TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}_cache_reward" \
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu --num_processes 2 \
    --main_process_port 29501 \
    scripts/train/finetune_moe_grpo.py "${COMMON_ARGS[@]}" \
    --sft-coef 0 \
    --wandb-run-name "cache-reward-local-${SLURM_JOB_ID}" \
    --save-dir "checkpoints/cache_reward_local_${SLURM_JOB_ID}" \
    > "cache_reward_local_${SLURM_JOB_ID}.log" 2>&1 &
pid_reward=$!

status=0
wait "$pid_sft" || status=1
wait "$pid_reward" || status=1
exit $status
