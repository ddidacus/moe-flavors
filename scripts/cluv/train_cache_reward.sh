#!/bin/bash
# Submit the cache_reward small-scale training job via cluv -- same as
# cache_sft (scripts/cluv/train_cache_sft.sh) but with --sft-coef 0: the
# SFT NLL loss term is fully skipped (GRPOTrainerWithSFT.compute_loss's
# `if self.sft_coef > 0` guard), so training is driven ONLY by the GRPO
# policy loss + cache-hit reward, no auxiliary supervised signal from the
# dataset's own ground-truth completions. Isolates what the cache reward
# alone does to the policy, without cache_sft's SFT term pulling it back
# toward the base model's original completions.
#
# Same context-length/batch/step budget as cache_sft (see that script's
# comment for the smoke-test rationale): prompt-len/completion-len =
# 1024/1024 (filtered, not truncated), batch-size=16, no gradient
# accumulation, 200 steps, max-samples 3200 = 200 x 16.
# ~7.8h wall-clock estimated on 4 GPUs. Saves to checkpoints/cache_reward_<CLUSTER>.
#
# Usage: CLUSTER=fir bash scripts/cluv/train_cache_reward.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/train/finetune_moe_grpo.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 3200 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --save-total-limit 3 --resume \
    --batch-size 16 --gradient-accumulation-steps 1 --num-steps 200 \
    --num-generations 8 --temperature 1.0 --rl-coef 2.0 --sft-coef 0 --beta 0.08 \
    --cache-size 4 --cache-layer -1 --cache-experts-per-token 2 --cache-topk --soft-cache \
    --eval-ppl-every 10 \
    --wandb-run-name "cache-reward-${CLUSTER}" --save-dir "checkpoints/cache_reward_${CLUSTER}"
echo "cache_reward -> submitted to $CLUSTER"
