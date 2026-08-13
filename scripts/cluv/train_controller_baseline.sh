#!/bin/bash
# Submit the controller_baseline small-scale training job via cluv (see
# scripts/train_small_scale.sh for the config this mirrors, and
# scripts/train/run_controller.sh for the mila/sbatch equivalent).
#
# batch-size=16, no gradient accumulation, 200 steps, max-samples=3200 --
# same unified budget as cache_sft/cache_reward/melinoe (see
# train_cache_sft.sh's comment for the smoke-test rationale). This script
# alone would comfortably fit 2048+2048 (scripts/train/smoke_test_context.sh
# measured ~69GB/80GB peak), but 1024+1024 is used for consistency with
# cache_sft/cache_reward, which is the binding constraint across the four
# variants (GRPO's on-policy generation is the most memory-hungry, and only
# just fits an A100L/H100 at 1024+1024, batch=16). Prompts are FILTERED to
# prompt-len, not truncated (see src.nemotron_data.sample_filtered_prompts).
# Saves to checkpoints/controller_baseline_<CLUSTER>_v2 -- the "_v2" avoids
# --resume picking up the OLD checkpoints/controller_baseline_<CLUSTER> run
# (step 32 under the old truncation-based data pipeline and hyperparameters),
# which would otherwise resume mid-run on stale config instead of starting
# a clean run under the new filtering/config.
#
# Usage: CLUSTER=fir bash scripts/cluv/train_controller_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/train/finetune_moe_controller.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 3200 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --resume \
    --batch-size 16 --gradient-accumulation-steps 1 --num-steps 200 \
    --cache-size 4 --cache-layer -1 --deliberation-cost 0.02 \
    --wandb-run-name "controller-baseline-${CLUSTER}-v2" --save-dir "checkpoints/controller_baseline_${CLUSTER}_v2"
echo "controller_baseline -> submitted to $CLUSTER"
