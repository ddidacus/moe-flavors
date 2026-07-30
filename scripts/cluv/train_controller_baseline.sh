#!/bin/bash
# Submit the controller_baseline small-scale training job via cluv (see
# scripts/train_small_scale.sh for the config this mirrors, and
# scripts/run_finetune_moe_controller.sh for the mila/sbatch equivalent).
# Saves to checkpoints/controller_baseline_<CLUSTER>.
#
# Usage: CLUSTER=fir bash scripts/cluv/train_controller_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/finetune_moe_controller.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 2000 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --resume \
    --batch-size 4 --gradient-accumulation-steps 4 --num-steps 32 \
    --cache-size 4 --cache-layer -1 --deliberation-cost 0.02 \
    --wandb-run-name "controller-baseline-${CLUSTER}" --save-dir "checkpoints/controller_baseline_${CLUSTER}"
echo "controller_baseline -> submitted to $CLUSTER"
