#!/bin/bash
# OLMoE-7B variant of train_sft_baseline.sh -- allenai/OLMoE-1B-7B-0125-Instruct
# instead of microsoft/Phi-tiny-MoE-instruct. See that script's comments for
# the shared dataset/scan/step-budget rationale (unchanged here).
#
# OLMoE-1B-7B-0125-Instruct: 64 experts (vs Phi-tiny-MoE's 16), top-8 routing
# (vs top-2), 16 layers (vs 32), ~7B total params / ~14GB bf16 weights (vs
# Phi-tiny-MoE's ~3.75B / ~7.5GB) -- roughly 2x the base-model memory
# footprint. model_type="olmoe" is NOT phimoe, so none of the fused-expert
# target_parameters special-casing in finetune_moe_sft.py applies -- this
# uses peft's plain/registered LoRA path unmodified, no code changes needed.
#
# --batch-size 8 --gradient-accumulation-steps 2 (halved from 16/1): plain
# SFT (no on-policy generation) is comparatively cheap even at 16/1 for
# Phi-tiny-MoE (job 418237 peaked at ~36GB/80GB), so this halving is a
# conservative, UNVALIDATED estimate for OLMoE's ~2x weight footprint, not
# an empirically confirmed setting -- there's no smoke test for OLMoE yet
# (cf. scripts/train/smoke_test_context.sh, which only covers Phi-tiny-MoE).
# Consider a short dry run (small --max-samples/--num-steps) before trusting
# the full 200-step budget.
#
# Usage: CLUSTER=tamia bash scripts/cluv/train_sft_baseline_olmoe.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/train/finetune_moe_sft.py \
    --model allenai/OLMoE-1B-7B-0125-Instruct \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 3200 --max-scan-per-split 300000 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --resume \
    --batch-size 8 --gradient-accumulation-steps 2 --num-steps 200 \
    --wandb-run-name "sft-baseline-olmoe-${CLUSTER}" --save-dir "checkpoints/sft_baseline_olmoe_${CLUSTER}"
echo "sft_baseline (OLMoE-7B) -> submitted to $CLUSTER"
