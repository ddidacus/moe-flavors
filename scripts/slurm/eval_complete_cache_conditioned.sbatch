#!/bin/bash
# Generic (non-cluv) SLURM job: runs ONE eval_complete_cache_conditioned.py
# invocation on a single GPU -- same as eval_complete.sbatch but for a
# prompt_conditioned checkpoint (swept across every cache size it was
# trained on). Edit --partition/--account/--gres below for your own
# SLURM setup.
#
# Requires >=40GB GPU VRAM (A100/A100L/L40S/H100 or equivalent).
#
# Usage:
#   sbatch scripts/slurm/eval_complete_cache_conditioned.sbatch \
#       --model microsoft/Phi-tiny-MoE-instruct \
#       --checkpoint-dir checkpoints/prompt_conditioned_tamia_200steps \
#       --cache-sizes 2,4,8 --cache-experts-per-token 2 --cache-topk
# (any extra args after the script name are forwarded to
# eval_complete_cache_conditioned.py)

#SBATCH --job-name=eval_complete_cc
#SBATCH --output=logs/slurm/eval_complete_cc_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# #SBATCH --partition=<your-gpu-partition>
# #SBATCH --account=<your-account>

set -euo pipefail
# SLURM copies this script to a spool dir before running it, so
# "dirname $0" does not point at the repo -- use the directory sbatch was
# invoked from instead.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs/slurm 2>/dev/null || true

source .venv/bin/activate
export HF_ALLOW_CODE_EVAL=1

python scripts/eval/eval_complete_cache_conditioned.py "$@"
