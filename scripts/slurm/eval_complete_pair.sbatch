#!/bin/bash
# Generic (non-cluv) SLURM job: runs TWO eval_complete.py (or
# eval_complete_cache_conditioned.py) invocations in parallel, each pinned
# to its own GPU -- the "2 scripts per job, 1 GPU + 4 CPUs each" packing
# scheme. Requests 2 GPUs total, 8 CPUs (4 per script). Edit
# --partition/--account/--gres below for your own SLURM setup.
#
# Each argument is a path to a small shell script containing one full
# `python scripts/eval/eval_complete.py ...` invocation (see
# scripts/slurm/submit_eval_sweep.sh, which generates these automatically).
# A single script path also works -- it just runs alone on GPU 0.
#
# Usage:
#   sbatch scripts/slurm/eval_complete_pair.sbatch <script1.sh> [<script2.sh>]

#SBATCH --job-name=eval_complete_pair
#SBATCH --output=logs/slurm/eval_complete_pair_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
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

bash scripts/eval/run_pair.sh "$@"
