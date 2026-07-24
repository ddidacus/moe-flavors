#!/bin/bash
#SBATCH --job-name=moe-flavors
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=h100:4
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# tamIA (Universite Laval, PAICE) -- H100 80GB nodes (H200 also available,
# swap gres to h200:4 if needed). Jobs must use all 4 GPUs on every node
# they're allocated, so we request the node exclusively rather than guess
# at exact CPU/mem-per-node figures. No internet on compute nodes (matches
# UV_OFFLINE/WANDB_MODE=offline default in pyproject.toml).

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
