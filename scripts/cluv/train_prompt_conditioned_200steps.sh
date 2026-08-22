#!/bin/bash
# Submit the prompt-conditioned cache-size GRPO job via cluv, at the SAME
# 200-step budget as the other baselines run on tamia for direct
# comparison (controller_baseline, melinoe_baseline, cache_sft,
# sft_baseline all use max-samples=3200, batch-size=16, num-steps=200 --
# see train_controller_baseline.sh/train_melinoe_baseline.sh/
# train_cache_sft.sh). scripts/cluv/train_prompt_conditioned.sh remains the
# separate, larger "full production" run (max-samples=10000, num-steps=625);
# this script is for parity with the sibling 200-step baselines instead.
#
# Trains scripts/train/train_prompt_conditioned.py: pure GRPO/DAPO (no SFT
# term), --beta 0.04, cache size conditioned via a "[CACHE_SIZE=X]" prompt
# prefix (X in {2,4,8}) parsed back out per-example at reward time.
#
# --max-samples 3200 is the BASE (unprefixed) prompt count for this
# script -- the dataset actually built has 3200 x 3 = 9600 rows (one row
# per cache size), matching the literal max-samples value used by every
# sibling baseline script even though (unlike them) that value isn't the
# final training-set size here -- see train_prompt_conditioned.py's own
# --max-samples help text. batch-size=16, no grad-accum, prompt-len/
# completion-len=1024/1024 mirror train_cache_reward.sh's proven-safe
# single-H100 memory profile (this script has no SFT NLL forward pass,
# architecturally identical to cache_reward on that axis).
# --cache-experts-per-token 2 --cache-topk mirrors the deterministic top-2
# routing cache_sft/cache_reward use.
#
# --time=1-00:00:00 matches melinoe_baseline/cache_sft's tamia budget;
# cache_sft (the closest architectural sibling: pure/soft-cache GRPO, same
# memory profile) needed 5h45m for 200 steps on tamia (job 418228), so 24h
# gives ample headroom even though this variant lacks --soft-cache's dense
# routing overhead and should be faster per step.
#
# Usage: CLUSTER=tamia bash scripts/cluv/train_prompt_conditioned_200steps.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/train/train_prompt_conditioned.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 3200 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --save-total-limit 3 --resume \
    --batch-size 16 --gradient-accumulation-steps 1 --num-steps 200 \
    --num-generations 8 --temperature 1.0 --beta 0.04 \
    --cache-layer -1 --cache-experts-per-token 2 --cache-topk \
    --prompt-cache-sizes 2,4,8 \
    --eval-ppl-every 10 --eval-hitrate-every 10 \
    --wandb-run-name "prompt-conditioned-${CLUSTER}-200steps" --save-dir "checkpoints/prompt_conditioned_${CLUSTER}_200steps"
echo "prompt_conditioned (200 steps) -> submitted to $CLUSTER"
