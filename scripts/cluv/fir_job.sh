#!/bin/bash
#SBATCH --job-name=moe-flavors
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=h100:4
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Fir (successor to Cedar) -- H100 80GB nodes, liquid-cooled. Requested
# exclusively (whole node) rather than guess at exact CPU/mem-per-node
# figures. Unrestricted internet on compute nodes (matches
# UV_OFFLINE=0/WANDB_MODE=online override in pyproject.toml).

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
