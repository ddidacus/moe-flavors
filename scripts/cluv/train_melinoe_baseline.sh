#!/bin/bash
# Submit the melinoe_baseline small-scale training job via cluv (see
# scripts/train_small_scale.sh for the config this mirrors, and
# scripts/run_finetune_moe_melinoe.sh for the mila/sbatch equivalent).
# MELINOE (Raje, Nayak & Joshi 2026, arXiv:2602.11192) -- cache-consistency
# + rank-margin loss, closer in spirit/cost to cache_sft (also a cache-aware
# RL-adjacent objective on top of the same base model) than to the plain
# SFT/controller baselines, so it borrows cache_sft's batch/accum/step/lr
# budget rather than sft_baseline's. LoRA rank/alpha are left at the
# paper's own defaults (r=32/alpha=16, see finetune_moe_melinoe.py) since
# --init-adapter isn't used here (fresh run, no cross-checkpoint shape
# constraint). Saves to checkpoints/melinoe_baseline_<CLUSTER>.
#
# Usage: CLUSTER=fir bash scripts/cluv/train_melinoe_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/finetune_moe_melinoe.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 2000 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 100 --resume \
    --batch-size 8 --gradient-accumulation-steps 2 --num-steps 250 \
    --cache-size 4 --cache-layer -1 \
    --wandb-run-name "melinoe-baseline-${CLUSTER}" --save-dir "checkpoints/melinoe_baseline_${CLUSTER}"
echo "melinoe_baseline -> submitted to $CLUSTER"
