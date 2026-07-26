#!/bin/bash
#SBATCH --job-name=moe-flavors
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Generic fallback job script (cluv default: scripts/job.sh), used for any
# cluster without its own `job_script_path` entry in pyproject.toml (today:
# mila, killarney, trillium, trillium-gpu). --gres pins A100L specifically
# (matching the mila run_finetune_moe_*.sh scripts' proven config) --
# without a GPU type constraint, Slurm happily assigns pre-Ampere GPUs
# (V100, RTX8000) that mila also has, which crash on bf16: "no kernel image
# is available for execution on the device" (V100 has no bf16 hardware
# support at all) or silent dtype-mismatch RuntimeErrors partway through
# training on RTX8000. --cpus-per-task/--mem match the same proven config
# -- without them Slurm's bare per-job default (1 cpu, 2G mem for the whole
# job) OOMs almost immediately. NOTE: --gres=gpu:a100l:4 is mila-specific;
# if this fallback ever actually gets used on killarney/trillium, it needs
# its own scripts/cluv/<cluster>_job.sh + job_script_path entry instead --
# see scripts/cluv/README.md.

source scripts/cluv/common.sh
