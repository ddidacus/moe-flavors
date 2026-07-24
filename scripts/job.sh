#!/bin/bash
#SBATCH --job-name=moe-flavors
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Generic fallback job script (cluv default: scripts/job.sh), used for any
# cluster without its own `job_script_path` entry in pyproject.toml (today:
# mila, killarney, trillium, trillium-gpu). Does not pin a GPU model/
# whole-node request, since that varies per cluster. --cpus-per-task/--mem
# match the mila run_finetune_moe_*.sh scripts' proven values for a 4-GPU
# accelerate job -- without them Slurm's bare per-job default (1 cpu, 2G
# mem for the whole job) OOMs almost immediately. Add a dedicated
# scripts/cluv/<cluster>_job.sh + job_script_path entry for any cluster you
# actually run on -- see scripts/cluv/README.md.

source scripts/cluv/common.sh
