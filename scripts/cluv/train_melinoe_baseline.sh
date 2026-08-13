#!/bin/bash
# Submit the melinoe_baseline small-scale training job via cluv (see
# scripts/train_small_scale.sh for the config this mirrors, and
# scripts/train/run_melinoe.sh for the mila/sbatch equivalent).
# MELINOE (Raje, Nayak & Joshi 2026, arXiv:2602.11192) -- cache-consistency
# + rank-margin loss, closer in spirit/cost to cache_sft (also a cache-aware
# RL-adjacent objective on top of the same base model) than to the plain
# SFT/controller baselines. LoRA rank/alpha are left at the paper's own
# defaults (r=32/alpha=16, see finetune_moe_melinoe.py) since --init-adapter
# isn't used here (fresh run, no cross-checkpoint shape constraint).
#
# batch-size=16, no gradient accumulation, 200 steps, max-samples=3200 --
# same unified budget as cache_sft/cache_reward/controller_baseline (see
# train_cache_sft.sh's comment for the smoke-test rationale). This script
# alone would comfortably fit 2048+2048 (scripts/train/smoke_test_context.sh
# measured ~73GB/80GB peak), but 1024+1024 is used for consistency with
# cache_sft/cache_reward, which is the binding constraint across the four
# variants. Prompts are FILTERED to prompt-len, not truncated (see
# src.nemotron_data.sample_filtered_prompts). Saves to
# checkpoints/melinoe_baseline_<CLUSTER>_v2 -- the "_v2" avoids --resume
# picking up the OLD checkpoints/melinoe_baseline_<CLUSTER> run (already at
# global_step==max_steps under the old truncation-based data pipeline and
# hyperparameters), which would otherwise silently no-op instead of
# actually retraining under the new filtering/config.
#
# Usage: CLUSTER=fir bash scripts/cluv/train_melinoe_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/train/finetune_moe_melinoe.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 3200 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --resume \
    --batch-size 16 --gradient-accumulation-steps 1 --num-steps 200 \
    --cache-size 4 --cache-layer -1 \
    --wandb-run-name "melinoe-baseline-${CLUSTER}-v2" --save-dir "checkpoints/melinoe_baseline_${CLUSTER}_v2"
echo "melinoe_baseline -> submitted to $CLUSTER"
