#!/bin/bash
#SBATCH --job-name=moe-flavors
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Generic fallback job script (cluv default: scripts/job.sh), used for any
# cluster without its own `job_script_path` entry in pyproject.toml. Does
# not pin a GPU model/whole-node request, since that varies per cluster.
# Add a dedicated scripts/cluv/<cluster>_job.sh + job_script_path entry
# for any cluster you actually run on -- see scripts/cluv/README.md.

source "$(dirname "${BASH_SOURCE[0]}")/cluv/common.sh"
