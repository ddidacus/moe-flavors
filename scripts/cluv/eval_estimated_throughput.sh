#!/bin/bash
# Submits scripts/eval/eval_estimated_throughput.py via cluv, on a job
# script pinned to H100 on BOTH clusters (mila_h100_job.sh here, tamia's own
# default tamia_job.sh already is H100) -- see mila_h100_job.sh's comment:
# this eval measures wall-clock latency, so every variant/model in the
# sweep must run on the same GPU model to be comparable. V100 (mila's other
# common GPU) was smoke-tested and fails outright: torch has no compiled
# kernel for its compute capability 7.0 ("no kernel image is available for
# execution on the device").
#
# MODEL selects the base model + its matching cache config (mirrors the
# --cache-size/--cache-experts-per-token used at training time in
# train_cache_reward.sh / train_cache_reward_olmoe.sh):
#   phi    microsoft/Phi-tiny-MoE-instruct,  cache-size 4,  experts/tok 2
#   olmoe  allenai/OLMoE-1B-7B-0125-Instruct, cache-size 16, experts/tok 8
#
# VARIANT is a free-form label (passed through to --variant; only "base"
# and "temporal_moe" are special-cased by build_variant_model, everything
# else takes the generic LoRA-adapter path) -- pass CHECKPOINT_DIR for
# every non-"base" variant.
#
# Usage:
#   CLUSTER=tamia MODEL=phi VARIANT=base \
#       bash scripts/cluv/eval_estimated_throughput.sh
#   CLUSTER=tamia MODEL=olmoe VARIANT=cache_reward \
#       CHECKPOINT_DIR=checkpoints/cache_reward_olmoe_tamia \
#       bash scripts/cluv/eval_estimated_throughput.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|mila>}"
MODEL="${MODEL:?set MODEL=<phi|olmoe>}"
VARIANT="${VARIANT:?set VARIANT=<base|sft_baseline|cache_reward|controller_baseline|melinoe_baseline|prompt_conditioned>}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

case "$MODEL" in
    phi)
        MODEL_NAME="microsoft/Phi-tiny-MoE-instruct"
        CACHE_SIZE=4
        EXPERTS_PER_TOKEN=2
        ;;
    olmoe)
        MODEL_NAME="allenai/OLMoE-1B-7B-0125-Instruct"
        CACHE_SIZE=16
        EXPERTS_PER_TOKEN=8
        ;;
    *) echo "MODEL must be 'phi' or 'olmoe', got '$MODEL'" >&2; exit 1 ;;
esac

ckpt_args=()
if [ -n "$CHECKPOINT_DIR" ]; then
    ckpt_args=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

if [ "$CLUSTER" = "tamia" ]; then
    JOB_SCRIPT="scripts/cluv/tamia_job.sh"  # already H100 by default
else
    JOB_SCRIPT="scripts/cluv/mila_h100_job.sh"
fi

cluv submit --autocommit "$CLUSTER" "$JOB_SCRIPT" --time=02:00:00 -- \
    python scripts/eval/eval_estimated_throughput.py \
    --model "$MODEL_NAME" --variant "$VARIANT" "${ckpt_args[@]}" \
    --cache-size "$CACHE_SIZE" --cache-layer -1 \
    --cache-experts-per-token "$EXPERTS_PER_TOKEN" --cache-topk \
    --num-expert-load-trials 1024 --num-eval-prompts 256 \
    --out-dir "evals/${MODEL_NAME##*/}/eval_estimated_throughput"
echo "eval_estimated_throughput (model=$MODEL variant=$VARIANT) -> submitted to $CLUSTER"
