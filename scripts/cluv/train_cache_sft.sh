#!/bin/bash
# Submit the cache_sft small-scale training job via cluv (see
# scripts/train_small_scale.sh for the config this mirrors, and
# scripts/run_finetune_moe_grpo.sh for the mila/sbatch equivalent).
# ~9.9h wall-clock estimated on 4 GPUs -- overrides the 3h pyproject.toml
# default walltime. Saves to checkpoints/cache_sft_<CLUSTER>.
#
# Usage: CLUSTER=fir bash scripts/cluv/train_cache_sft.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/finetune_moe_grpo.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 2000 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --resume \
    --batch-size 8 --gradient-accumulation-steps 2 --num-steps 250 --num-epochs 10 \
    --num-generations 8 --temperature 1.0 --rl-coef 2.0 --sft-coef 0.5 --beta 0.08 \
    --cache-size 4 --cache-layer -1 --cache-experts-per-token 2 --cache-topk --soft-cache \
    --eval-ppl-every 10 \
    --wandb-run-name "cache-sft-${CLUSTER}" --save-dir "checkpoints/cache_sft_${CLUSTER}"
echo "cache_sft -> submitted to $CLUSTER"
