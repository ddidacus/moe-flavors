#!/bin/bash
# Mila cluster (plain sbatch, NOT cluv): runs ONE eval_consistency_robustness.py
# invocation on a single GPU. This eval is deliberately cheap -- no
# lm-eval-harness, no 1024-prompt generation sweep -- so a single GPU for
# a couple of hours is enough per (model, variant) combo.
#
# Usage:
#   sbatch scripts/slurm/eval_consistency_robustness.sbatch \
#       --model microsoft/Phi-tiny-MoE-instruct --variant base
#   sbatch scripts/slurm/eval_consistency_robustness.sbatch \
#       --model allenai/OLMoE-1B-7B-0125-Instruct --variant cache_reward \
#       --checkpoint-dir ddidacus/olmoe-cache-reward --cache-layer -1
# (any extra args after the script name are forwarded to
# eval_consistency_robustness.py)

#SBATCH --job-name=eval_consistency
#SBATCH --output=logs/slurm/eval_consistency_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs/slurm 2>/dev/null || true

source .venv/bin/activate
export HF_ALLOW_CODE_EVAL=1
# Long partition's node pool includes old Volta/Turing GPUs (V100, RTX8000)
# our torch build has no compiled kernels for ("no kernel image is
# available") -- pin to l40s (Ada, plentiful on this cluster) above.
export HF_HUB_DISABLE_XET=1
# The REAL "Disk quota exceeded" cause: $HOME is a 100GB-quota Lustre
# mount and is already over quota (104.9/100GB used) independent of any
# one job here -- every download into the default ~/.cache/huggingface
# fails immediately regardless of GPU/Xet fixes. Redirect the HF cache to
# $SCRATCH (beegfs, hundreds of TB free, no comparable per-user quota).
export HF_HOME="$SCRATCH/hf_cache_eval_consistency"
mkdir -p "$HF_HOME"

python scripts/eval/eval_consistency_robustness.py "$@"
