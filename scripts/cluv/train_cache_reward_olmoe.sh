#!/bin/bash
# OLMoE-7B variant of train_cache_reward.sh -- same as cache_sft_olmoe.sh
# (scripts/cluv/train_cache_sft_olmoe.sh) but with --sft-coef 0: the SFT NLL
# loss term is fully skipped (GRPOTrainerWithSFT.compute_loss's
# `if self.sft_coef > 0` guard), so training is driven ONLY by the GRPO
# policy loss + cache-hit reward, no auxiliary supervised signal from the
# dataset's own ground-truth completions -- isolates what the cache reward
# alone does to the policy. allenai/OLMoE-1B-7B-0125-Instruct instead of
# microsoft/Phi-tiny-MoE-instruct; see train_cache_sft_olmoe.sh's comments
# for the shared 64-expert/top-8/layer-8/cache-size-16 rationale.
#
# --batch-size 8 --gradient-accumulation-steps 2 (vs cache_sft_olmoe's 4/4):
# same 2x ratio Phi-tiny-MoE's own cache_reward/cache_sft pair uses (16/1
# vs 8/2) -- dropping the SFT NLL forward pass frees enough memory to
# double the batch at the same effective size. Still a conservative,
# UNVALIDATED estimate for OLMoE's ~2x weight footprint -- no OLMoE smoke
# test exists yet.
#
# Usage: CLUSTER=tamia bash scripts/cluv/train_cache_reward_olmoe.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/train/finetune_moe_grpo.py \
    --model allenai/OLMoE-1B-7B-0125-Instruct \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 3200 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --save-total-limit 3 --resume \
    --batch-size 8 --gradient-accumulation-steps 2 --num-steps 200 \
    --num-generations 8 --temperature 1.0 --rl-coef 2.0 --sft-coef 0 --beta 0.08 \
    --cache-size 16 --cache-layer -1 --cache-experts-per-token 8 --cache-topk --soft-cache \
    --eval-ppl-every 10 \
    --wandb-run-name "cache-reward-olmoe-${CLUSTER}" --save-dir "checkpoints/cache_reward_olmoe_${CLUSTER}"
echo "cache_reward (OLMoE-7B) -> submitted to $CLUSTER"
