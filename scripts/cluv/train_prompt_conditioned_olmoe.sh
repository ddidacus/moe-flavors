#!/bin/bash
# OLMoE-7B variant of train_prompt_conditioned_200steps.sh --
# allenai/OLMoE-1B-7B-0125-Instruct instead of microsoft/Phi-tiny-MoE-instruct.
# See that script's comments for the shared dataset/scan/step-budget
# rationale (unchanged here).
#
# OLMoE-1B-7B-0125-Instruct: 64 experts (vs 16), top-8 routing (vs top-2),
# 16 layers (vs 32), ~2x the base-model weight footprint of Phi-tiny-MoE.
# --cache-layer -1 resolves to layer 8 (16 // 2) instead of 16.
# --cache-experts-per-token 8 matches OLMoE's OWN native top-8 routing.
# --prompt-cache-sizes 8,16,32 scales the Phi-tiny-MoE config's 2,4,8 by
# the same 4x ratio as num_experts (16->64), preserving the same
# 12.5%/25%/50%-of-total-experts fractions.
#
# --batch-size 8 --gradient-accumulation-steps 2 (halved from this script's
# 16/1 for Phi-tiny-MoE -- architecturally identical to cache_reward, the
# cheapest of the four GRPO-like objectives since it has no SFT NLL forward
# pass and no --soft-cache dense-routing overhead): a conservative,
# UNVALIDATED estimate for OLMoE's ~2x weight footprint -- there's no
# OLMoE smoke test yet. Consider a short dry run (small --max-samples/
# --num-steps) before trusting the full 200-step budget.
#
# --num_processes 1: job 422274 hit the same NCCL collective-operation
# timeout as the Phi-tiny-MoE variant at --num_processes 4 (see
# train_prompt_conditioned_200steps.sh) -- confirming the hang is specific
# to this script, not model-dependent. Only --num_processes 1 (no
# --multi_gpu) is proven reliable for it.
#
# Usage: CLUSTER=tamia bash scripts/cluv/train_prompt_conditioned_olmoe.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --num_processes 1 \
    scripts/train/train_prompt_conditioned.py \
    --model allenai/OLMoE-1B-7B-0125-Instruct \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 3200 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --save-total-limit 3 --resume \
    --batch-size 8 --gradient-accumulation-steps 2 --num-steps 200 \
    --num-generations 8 --temperature 1.0 --beta 0.04 \
    --cache-layer -1 --cache-experts-per-token 8 --cache-topk \
    --prompt-cache-sizes 8,16,32 \
    --eval-ppl-every 10 --eval-hitrate-every 10 \
    --wandb-run-name "prompt-conditioned-olmoe-${CLUSTER}" --save-dir "checkpoints/prompt_conditioned_olmoe_${CLUSTER}"
echo "prompt_conditioned (OLMoE-7B) -> submitted to $CLUSTER"
