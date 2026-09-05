#!/bin/bash
# Resubmits the 4 OLMoE combos that hit CUDA OOM in the original
# eval_reuse_semantic_drift sweep (KL-divergence step held two full
# (B, S, vocab) float32 tensors simultaneously). Fixed in
# eval_reuse_semantic_drift.py (streamed per-sequence KL, freed
# intermediates promptly); also halves batch size for OLMoE as a margin.
#
# Usage: bash scripts/slurm/resubmit_eval_reuse_semantic_drift_olmoe.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/slurm

OLMOE="allenai/OLMoE-1B-7B-0125-Instruct"

combos=(
    "$OLMOE|sft|ddidacus/olmoe-sft"
    "$OLMOE|cache_reward|ddidacus/olmoe-cache-reward"
    "$OLMOE|controller|ddidacus/olmoe-controller"
    "$OLMOE|melinoe|ddidacus/olmoe-melinoe"
)

for combo in "${combos[@]}"; do
    IFS='|' read -r model variant ckpt <<< "$combo"
    args=(--model "$model" --variant "$variant" --batch-size 8)
    if [ -n "$ckpt" ]; then
        args+=(--checkpoint-dir "$ckpt")
    fi
    echo "Submitting: ${args[*]}"
    sbatch --partition=long scripts/slurm/eval_reuse_semantic_drift.sbatch "${args[@]}"
done
