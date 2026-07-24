#!/bin/bash
#SBATCH --job-name=moe-flavors
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=h100:4
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Nibi (successor to Graham) -- H100 80GB nodes (also has AMD MI300A nodes,
# not used here). Requested exclusively (whole node) rather than guess at
# exact CPU/mem-per-node figures. Unrestricted internet on compute nodes
# (matches the UV_OFFLINE=0 override in pyproject.toml) -- wandb is still
# forced off for cluv jobs regardless, see common.sh.

source scripts/cluv/common.sh
