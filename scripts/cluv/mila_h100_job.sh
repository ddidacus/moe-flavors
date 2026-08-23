#!/bin/bash
#SBATCH --job-name=moe-flavors-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=short-unkillable
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# H100-pinned job script for mila, used ONLY for evals that need the exact
# same GPU model as tamia (which only has H100) for comparable wall-clock
# timing -- eval_estimated_throughput.py's whole premise is a latency
# measurement, so mixing GPU models across the sweep (e.g. mila's default
# scripts/cluv/job.sh, which pins A100L) would make the mila and tamia
# numbers incomparable. mila's H100s only exist in the short-unkillable
# partition, whose QOS enforces a 4-GPU minimum (sacctmgr show qos
# short-partition: MinTRES cpu=4,gres/gpu=4) even though the eval script
# itself is single-GPU/single-process -- same 3-of-4-GPUs-idle tradeoff
# tamia_job.sh already accepts for the single-GPU training jobs.
#
# NOT the default job_script_path for mila (that stays scripts/cluv/job.sh,
# A100L) -- pass this explicitly: cluv submit mila scripts/cluv/mila_h100_job.sh -- ...

source scripts/cluv/common.sh
