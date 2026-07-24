#!/bin/bash
#SBATCH --job-name=moe-flavors
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=l40s:4
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Vulcan (University of Alberta / Amii) -- L40S 48GB nodes, 4 GPUs + 64
# CPU cores per node (confirmed via alliancecan.ca). Requested exclusively
# to grab the full node. L40S has less VRAM than the A100L used on Mila
# (48GB vs 80GB) -- lower --batch-size / raise --gradient-accumulation-steps
# if a job OOMs here.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
