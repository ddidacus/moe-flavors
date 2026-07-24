#!/bin/bash
#SBATCH --job-name=moe-flavors
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=a100:4
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Narval (ETS Montreal) -- A100 40GB nodes (older cluster, no H100/H200;
# good for runs that don't need H100-class throughput). Mila's allocation
# here uses account def-bengioy (see pyproject.toml env), and access
# depends on supervisor affiliation -- confirm you have narval access
# before relying on this. Requested exclusively (whole node, 4 GPUs).
# No internet on compute nodes.

source scripts/cluv/common.sh
