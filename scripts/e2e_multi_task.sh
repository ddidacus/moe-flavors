#!/bin/bash
#SBATCH --job-name=multi_task_moe
#SBATCH --output=multi_task_moe_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:4
#SBATCH --partition=long
#SBATCH --time=12:00:00

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1

SAVE_DIR="${RESUME_CKPT_DIR:-checkpoints/multi_task_temporal_${SLURM_JOB_ID:-local}}"

# ── Training (skip if checkpoint already exists) ──

FINAL_CKPT=$(ls -d "$SAVE_DIR"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)

if [ -n "$FINAL_CKPT" ]; then
    echo "Found existing checkpoint: $FINAL_CKPT — skipping training."
else
    accelerate launch --multi_gpu --num_processes=4 scripts/multi_task_moe.py \
        --moe-type temporal \
        --model Qwen/Qwen3-0.6B \
        --num-experts 16 \
        --top-k 4 \
        --ratio-loss-N 3 3 4 4 5 5 6 6 7 7 7 8 8 9 9 10 10 11 11 11 12 12 13 13 14 14 15 15 \
        --ratio-loss-alpha 0.03 \
        --entropy-threshold 0.1 \
        --entropy-alpha 0.05 \
        --entropy-warmup-steps 500 \
        --seq-len 4096 \
        --batch-size 16 \
        --dataset-splits stem chat math code \
        --num-samples 1000000 \
        --num-steps 30000 \
        --lr 8e-4 \
        --log-every 10 \
        --eval-every 5000 \
        --seed 42 \
        --wandb-project moe-multi-task \
        --wandb-run-name multi-task-temporal-16e-k4 \
        --save-dir "$SAVE_DIR" \
        --save-every 10000

    FINAL_CKPT=$(ls -d "$SAVE_DIR"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
fi

if [ -z "$FINAL_CKPT" ]; then
    echo "ERROR: No checkpoint found in $SAVE_DIR"
    exit 1
fi

# ── Analysis ──

echo "Analyzing checkpoint: $FINAL_CKPT"

python scripts/analyze_expert_specialization.py \
    --checkpoint-dir "$FINAL_CKPT" \
    --num-samples 10000 \
    --max-sample-len 1024 \
    --data-offset 50000 \
    --batch-size 32
