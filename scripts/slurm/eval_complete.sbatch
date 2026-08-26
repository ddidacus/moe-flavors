#!/bin/bash
# Generic (non-cluv) SLURM job: runs ONE eval_complete.py invocation on a
# single GPU. Portable across clusters -- edit --partition/--account/--gres
# below for your own SLURM setup (this repo's own cluster uses cluv, see
# scripts/cluv/, but that tooling is specific to that environment).
#
# Requires >=40GB GPU VRAM (A100/A100L/L40S/H100 or equivalent).
#
# Usage:
#   sbatch scripts/slurm/eval_complete.sbatch \
#       --model microsoft/Phi-tiny-MoE-instruct --variant base
#   sbatch scripts/slurm/eval_complete.sbatch \
#       --model allenai/OLMoE-1B-7B-0125-Instruct --variant cache_reward \
#       --checkpoint-dir checkpoints/cache_reward_olmoe_tamia \
#       --cache-size 16 --cache-experts-per-token 8 --cache-topk
# (any extra args after the script name are forwarded to eval_complete.py)

#SBATCH --job-name=eval_complete
#SBATCH --output=logs/slurm/eval_complete_%j.out
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

python scripts/eval/eval_complete.py "$@"
