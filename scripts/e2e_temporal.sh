#!/bin/bash
#SBATCH --job-name=e2e_temporal
#SBATCH --output=e2e_temporal_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:4
#SBATCH --partition=long
#SBATCH --time=6:00:00

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1
# export PYTHONPATH="/network/scratch/d/diego.calanzone/moe-playground/stubs:${PYTHONPATH:-}"

SAVE_DIR="${RESUME_CKPT_DIR:-checkpoints/temporal_${SLURM_JOB_ID:-local}}"

# ── Training (skip if checkpoint already exists) ──

FINAL_CKPT=$(ls -d "$SAVE_DIR"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)

if [ -n "$FINAL_CKPT" ]; then
    echo "Found existing checkpoint: $FINAL_CKPT — skipping training."
else
    accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
        --moe-type temporal \
        --model Qwen/Qwen3-0.6B \
        --num-experts 4 \
        --top-k 1 \
        --ratio-loss-N 3 3 4 4 5 5 6 6 7 7 7 8 8 9 9 10 10 11 11 11 12 12 13 13 14 14 15 15 \
        --ratio-loss-alpha 0.03 \
        --entropy-threshold 0.1 \
        --entropy-alpha 0.05 \
        --entropy-warmup-steps 500 \
        --seq-len 256 \
        --batch-size 16 \
        --dataset-splits math \
        --num-samples 1000000 \
        --num-steps 30000 \
        --lr 8e-4 \
        --log-every 10 \
        --eval-every 50000 \
        --harness-limit 8 \
        --seed 42 \
        --wandb-project moe-chunking-poc \
        --wandb-run-name temporal-moe-N-9 \
        --save-dir "$SAVE_DIR" \
        --save-every 10000

    FINAL_CKPT=$(ls -d "$SAVE_DIR"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
fi

if [ -z "$FINAL_CKPT" ]; then
    echo "ERROR: No checkpoint found in $SAVE_DIR"
    exit 1
fi

echo "Evaluating checkpoint: $FINAL_CKPT"

python scripts/eval_harness.py \
    --checkpoint-dir "$FINAL_CKPT" \
    --batch-size auto \
    --output-dir "$FINAL_CKPT/eval_results" \
    --wandb-project moe-chunking-poc
