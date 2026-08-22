#!/bin/bash
# OLMoE-7B variant of train_cache_sft.sh -- allenai/OLMoE-1B-7B-0125-Instruct
# instead of microsoft/Phi-tiny-MoE-instruct. See that script's comments for
# the shared dataset/scan/step-budget rationale (unchanged here).
#
# OLMoE-1B-7B-0125-Instruct: 64 experts (vs 16), top-8 routing (vs top-2),
# 16 layers (vs 32), ~2x the base-model weight footprint of Phi-tiny-MoE.
# --cache-layer -1 resolves to layer 8 (16 // 2) instead of 16.
# --cache-experts-per-token 8 matches OLMoE's OWN native top-8 routing
# (Phi-tiny-MoE's top-2 default would misrepresent what the model actually
# does if left at 2). --cache-size 16 scales the Phi-tiny-MoE config's
# --cache-size 4 by the same 4x ratio as num_experts (16->64), preserving
# the same 25%-of-total-experts cache fraction.
#
# --batch-size 4 --gradient-accumulation-steps 4 (halved AGAIN from
# cache_sft's already-halved 8/2 for Phi-tiny-MoE, which itself only exists
# because batch=16 OOM'd a single H100 on job 407136): this is a
# conservative, UNVALIDATED estimate for OLMoE's ~2x weight footprint ON
# TOP of GRPO's already memory-hungry on-policy generation + --soft-cache
# dense routing -- there's no OLMoE smoke test yet. Strongly consider a
# short dry run (small --max-samples/--num-steps) before trusting the full
# 200-step budget; this combination (largest model x most memory-hungry
# objective) is the single highest OOM-risk script in this new batch.
#
# Usage: CLUSTER=tamia bash scripts/cluv/train_cache_sft_olmoe.sh
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
    --batch-size 4 --gradient-accumulation-steps 4 --num-steps 200 \
    --num-generations 8 --temperature 1.0 --rl-coef 2.0 --sft-coef 0.5 --beta 0.08 \
    --cache-size 16 --cache-layer -1 --cache-experts-per-token 8 --cache-topk --soft-cache \
    --eval-ppl-every 10 \
    --wandb-run-name "cache-sft-olmoe-${CLUSTER}" --save-dir "checkpoints/cache_sft_olmoe_${CLUSTER}"
echo "cache_sft (OLMoE-7B) -> submitted to $CLUSTER"
