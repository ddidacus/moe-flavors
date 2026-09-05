#!/bin/bash
# Mila cluster (plain sbatch, NOT cluv): runs ONE eval_reuse_semantic_drift.py
# invocation on a single GPU. Cheap: one greedy generation (shared, base-
# model-only) plus two forward passes per prompt batch, no lm-eval-harness.
#
# Usage:
#   sbatch scripts/slurm/eval_reuse_semantic_drift.sbatch \
#       --model microsoft/Phi-tiny-MoE-instruct --variant base
#   sbatch scripts/slurm/eval_reuse_semantic_drift.sbatch \
#       --model allenai/OLMoE-1B-7B-0125-Instruct --variant cache_reward \
#       --checkpoint-dir ddidacus/olmoe-cache-reward
# (any extra args after the script name are forwarded to
# eval_reuse_semantic_drift.py)

#SBATCH --job-name=eval_reuse_drift
#SBATCH --output=logs/slurm/eval_reuse_drift_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs/slurm 2>/dev/null || true

source .venv/bin/activate
export HF_ALLOW_CODE_EVAL=1
# Same fixes as eval_consistency_robustness.sbatch: long's node pool has
# GPUs our torch build has no kernels for (pin l40s), Xet download races
# under concurrent OLMoE downloads (disabled), and $HOME's Lustre quota
# blocks the default HF cache location entirely (redirect to $SCRATCH,
# already pre-warmed with OLMoE-1B-7B-0125-Instruct's weights).
export HF_HUB_DISABLE_XET=1
export HF_HOME="$SCRATCH/hf_cache_eval_consistency"
mkdir -p "$HF_HOME"

python scripts/eval/eval_reuse_semantic_drift.py "$@"
