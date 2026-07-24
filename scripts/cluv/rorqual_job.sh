#!/bin/bash
#SBATCH --job-name=moe-flavors
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=h100:4
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Rorqual (ETS Montreal, successor to Beluga) -- H100 80GB nodes. Requested
# exclusively (whole node) to match the other 4-GPU-per-node runs rather
# than guess at exact CPU/mem-per-node figures. No internet on compute
# nodes -- use `module load httpproxy` if you need live wandb/CometML
# instead of the offline default set in pyproject.toml.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
