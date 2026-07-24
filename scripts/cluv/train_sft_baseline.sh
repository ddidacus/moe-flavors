#!/bin/bash
# Submit the sft_baseline small-scale training job via cluv (see
# scripts/train_small_scale.sh for the config this mirrors, and
# scripts/run_finetune_moe_sft.sh for the mila/sbatch equivalent).
# Saves to checkpoints/sft_baseline_<CLUSTER> -- matches what
# scripts/cluv/eval_lm_harness.sh / eval_soft_cache.sh look up by default.
#
# Usage: CLUSTER=fir bash scripts/cluv/train_sft_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/finetune_moe_sft.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 2000 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --resume \
    --batch-size 4 --gradient-accumulation-steps 4 --num-steps 32 --num-epochs 10 \
    --wandb-run-name "sft-baseline-${CLUSTER}" --save-dir "checkpoints/sft_baseline_${CLUSTER}"
echo "sft_baseline -> submitted to $CLUSTER"
