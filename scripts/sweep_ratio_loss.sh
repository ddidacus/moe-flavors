#!/bin/bash
# Launches a grid sweep over ratio_loss_N and ratio_loss_alpha.
# Usage: bash scripts/sweep_ratio_loss.sh

RATIO_LOSS_NS=(3 6 9)
RATIO_LOSS_ALPHAS=(0.02 0.05 0.1)

for N in "${RATIO_LOSS_NS[@]}"; do
    for ALPHA in "${RATIO_LOSS_ALPHAS[@]}"; do
        JOB_NAME="tmoe_N${N}_a${ALPHA}"
        echo "Submitting ${JOB_NAME}"
        sbatch \
            --job-name="${JOB_NAME}" \
            --output="${JOB_NAME}_%j.out" \
            --cpus-per-task=16 \
            --mem=64G \
            --gres=gpu:80gb:4 \
            --partition=long \
            --time=12:00:00 \
            --wrap="
source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache

accelerate launch --multi_gpu --num_processes=4 scripts/moe_mixin_poc.py \
    --moe-type temporal \
    --model Qwen/Qwen3-0.6B \
    --num-experts 4 \
    --top-k 1 \
    --ratio-loss-N ${N} \
    --ratio-loss-alpha ${ALPHA} \
    --entropy-threshold 0.1 \
    --entropy-alpha 1.0 \
    --entropy-warmup-steps 0 \
    --seq-len 256 \
    --batch-size 32 \
    --dataset-splits code math \
    --num-samples 1000000 \
    --num-steps 100000 \
    --lr 1e-4 \
    --log-every 10 \
    --eval-every 50 \
    --seed 42 \
    --wandb-project moe-chunking-sweep
"
    done
done
