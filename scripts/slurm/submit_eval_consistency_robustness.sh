#!/bin/bash
# Submits eval_consistency_robustness.py for every available checkpoint of
# both backbones on the Mila cluster directly (plain sbatch, not cluv).
# One GPU, ~2h per job -- this eval is cheap (no lm-eval-harness rerun).
#
# Usage: bash scripts/slurm/submit_eval_consistency_robustness.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/slurm

PHI="microsoft/Phi-tiny-MoE-instruct"
OLMOE="allenai/OLMoE-1B-7B-0125-Instruct"

# model, variant, checkpoint-dir (empty for base)
combos=(
    "$PHI|base|"
    "$PHI|sft|ddidacus/phi-tiny-moe-sft"
    "$PHI|cache_reward|ddidacus/phi-tiny-moe-cache-reward"
    "$PHI|controller|ddidacus/phi-tiny-moe-controller"
    "$PHI|melinoe|ddidacus/phi-tiny-moe-melinoe"
    "$OLMOE|base|"
    "$OLMOE|sft|ddidacus/olmoe-sft"
    "$OLMOE|cache_reward|ddidacus/olmoe-cache-reward"
    "$OLMOE|controller|ddidacus/olmoe-controller"
    "$OLMOE|melinoe|ddidacus/olmoe-melinoe"
)

for combo in "${combos[@]}"; do
    IFS='|' read -r model variant ckpt <<< "$combo"
    args=(--model "$model" --variant "$variant")
    if [ -n "$ckpt" ]; then
        args+=(--checkpoint-dir "$ckpt")
    fi
    echo "Submitting: ${args[*]}"
    sbatch --partition=main scripts/slurm/eval_consistency_robustness.sbatch "${args[@]}"
done
