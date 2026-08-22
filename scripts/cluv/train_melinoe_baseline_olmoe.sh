#!/bin/bash
# OLMoE-7B variant of train_melinoe_baseline.sh --
# allenai/OLMoE-1B-7B-0125-Instruct instead of microsoft/Phi-tiny-MoE-instruct.
# See that script's comments for the shared dataset/scan/step-budget
# rationale (unchanged here). MELINOE (Raje, Nayak & Joshi 2026,
# arXiv:2602.11192) is also the original paper's own primary evaluation
# backbone, alongside Phi-3.5-MoE and Mixtral-8x7B -- this is the first
# time it's tested here against a model closer to that paper's own scale.
#
# OLMoE-1B-7B-0125-Instruct: 64 experts (vs 16), top-8 routing (vs top-2),
# 16 layers (vs 32), ~2x the base-model weight footprint of Phi-tiny-MoE.
# --cache-layer -1 resolves to layer 8 (16 // 2) instead of 16.
# --cache-size 16 scales the Phi-tiny-MoE config's --cache-size 4 by the
# same 4x ratio as num_experts (16->64), preserving the same 25%-of-total-
# experts cache fraction.
#
# --batch-size 8 --gradient-accumulation-steps 2 (halved from
# melinoe_baseline's 16/1 for Phi-tiny-MoE, which measured ~73GB/80GB peak
# at 2048+2048): a conservative, UNVALIDATED estimate for OLMoE's ~2x
# weight footprint -- there's no OLMoE smoke test yet. Consider a short
# dry run (small --max-samples/--num-steps) before trusting the full
# 200-step budget.
#
# Usage: CLUSTER=tamia bash scripts/cluv/train_melinoe_baseline_olmoe.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/train/finetune_moe_melinoe.py \
    --model allenai/OLMoE-1B-7B-0125-Instruct \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 3200 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --resume \
    --batch-size 8 --gradient-accumulation-steps 2 --num-steps 200 \
    --cache-size 16 --cache-layer -1 \
    --wandb-run-name "melinoe-baseline-olmoe-${CLUSTER}" --save-dir "checkpoints/melinoe_baseline_olmoe_${CLUSTER}"
echo "melinoe_baseline (OLMoE-7B) -> submitted to $CLUSTER"
