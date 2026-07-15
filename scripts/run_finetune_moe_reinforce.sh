#!/bin/bash
#SBATCH --job-name=rl_olmoe_cache
#SBATCH --output=rl_olmoe_cache_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

# --- Configuration ---
ALPHA="${ALPHA:-0.5}"        # weight of r^BC vs (1-ALPHA) for r^cache
TAU="${TAU:-0.2}"            # teacher fraction of the sampling mixture p_mix
CACHE_SIZE="${CACHE_SIZE:-16}"
CACHE_LAYER="${CACHE_LAYER:--1}"   # -1 = middle layer (8/16 for OLMoE)

SAVE_DIR="checkpoints/reinforce_olmoe_cache_mathcode_a${ALPHA}_c${CACHE_SIZE}"
RUN_NAME="reinforce-olmoe-cache-mathcode-a${ALPHA}-tau${TAU}-c${CACHE_SIZE}"

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    scripts/finetune_moe_reinforce.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split "math,code" \
    --max-samples 20000 \
    --prompt-len 512 \
    --gen-len 256 \
    --batch-size 64 \
    --gradient-accumulation-steps 1 \
    --num-epochs 1 \
    --num-steps 500 \
    --lr 1e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.03 \
    --lr-scheduler constant_with_warmup \
    --alpha "$ALPHA" \
    --mix-tau "$TAU" \
    --gamma 1.0 \
    --baseline-ema 0.9 \
    --cache-size "$CACHE_SIZE" \
    --cache-layer "$CACHE_LAYER" \
    --lora-r 16 \
    --lora-alpha 16 \
    --lora-dropout 0.0 \
    --log-every 1 \
    --eval-every 250 \
    --seed 42 \
    --wandb-project moe-cache-reinforce \
    --wandb-run-name "$RUN_NAME" \
    --save-dir "$SAVE_DIR" \
    --save-every 50 \
    --resume-from auto
