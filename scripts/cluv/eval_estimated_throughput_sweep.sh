#!/bin/bash
# Submits scripts/cluv/eval_estimated_throughput.sh for every (model,
# variant) combo -- see evals/EVAL_STATUS.md for the checkpoint-readiness
# table this mirrors. All 12 combos (5 variants + base, x2 models) are
# checkpoint-ready as of 2026-08-23.
#
# Usage: CLUSTER=tamia bash scripts/cluv/eval_estimated_throughput_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|mila>}"

submit() {
    local model=$1 variant=$2 ckpt=$3
    echo "=== $model / $variant ==="
    CLUSTER="$CLUSTER" MODEL="$model" VARIANT="$variant" CHECKPOINT_DIR="$ckpt" \
        bash scripts/cluv/eval_estimated_throughput.sh
}

# --- Phi-tiny-MoE ---
submit phi base ""
submit phi sft_baseline "checkpoints/sft_baseline_tamia"
submit phi cache_reward "checkpoints/cache_reward_tamia"
submit phi controller_baseline "checkpoints/controller_baseline_tamia_v2"
submit phi melinoe_baseline "checkpoints/melinoe_baseline_tamia_v2"
submit phi prompt_conditioned "checkpoints/prompt_conditioned_tamia_200steps"

# --- OLMoE-7B ---
submit olmoe base ""
submit olmoe sft_baseline "checkpoints/sft_baseline_olmoe_tamia"
submit olmoe cache_reward "checkpoints/cache_reward_olmoe_tamia"
submit olmoe controller_baseline "checkpoints/controller_baseline_olmoe_tamia"
submit olmoe melinoe_baseline "checkpoints/melinoe_baseline_olmoe_tamia"
submit olmoe prompt_conditioned "checkpoints/prompt_conditioned_olmoe_tamia"
