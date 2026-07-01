#!/bin/bash
#SBATCH --job-name=temporal_moe
#SBATCH --output=temporal_moe_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:80gb:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

# --- Configuration ---
# Set LEARNABLE_N=1 to enable the learnable-N variant
LEARNABLE_N="${LEARNABLE_N:-0}"

if [ "$LEARNABLE_N" = "1" ]; then
    SAVE_DIR="checkpoints/temporal_moe_64e_k8_learnable_N"
    RUN_NAME="temporal-moe-qwen0.6b-64e-k8-learnableN"
    EXTRA_ARGS="--learnable-N"
else
    SAVE_DIR="checkpoints/temporal_moe_64e_k8"
    RUN_NAME="temporal-moe-qwen0.6b-64e-k8"
    EXTRA_ARGS=""
fi

accelerate launch --multi_gpu --num_processes=4 scripts/train_moe.py \
    --moe-type temporal \
    --model Qwen/Qwen3-0.6B \
    --num-experts 64 \
    --top-k 8 \
    --expert-dim 32 \
    --ratio-loss-N 8 \
    --ratio-loss-alpha 0.03 \
    --entropy-threshold 0.1 \
    --entropy-alpha 0.05 \
    --entropy-warmup-steps 500 \
    --seq-len 8192 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --dataset ddidacus/nemotron-moe-exam \
    --num-epochs 1 \
    --lr 4e-5 \
    --weight-decay 0.1 \
    --warmup-ratio 0.05 \
    --lr-scheduler cosine \
    --log-every 1 \
    --eval-every 50 \
    --seed 42 \
    --wandb-project moe-chunking-poc \
    --wandb-run-name "$RUN_NAME" \
    --save-dir "$SAVE_DIR" \
    --save-every 100 \
    --resume-from auto \
    $EXTRA_ARGS
