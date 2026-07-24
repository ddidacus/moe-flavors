#!/bin/bash
# Shared body for scripts/cluv/*_job.sh. Each cluster's job script sets its
# own #SBATCH resource header (GPU type/count for that cluster) and then
# sources this file, which sets up the environment and execs the program
# passed after `--` by `cluv submit <cluster> -- <program> [args...]`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-$HOME}/cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

# Compute nodes on these clusters have no internet access, so wandb (even
# in "offline" mode) is more trouble than it's worth -- force it off and
# rely on the SLURM stdout log (%x_%j.out) instead. Overrides any
# WANDB_MODE set via [tool.cluv.env] / [tool.cluv.clusters.*].env.
export WANDB_MODE=disabled

if [ "$#" -eq 0 ]; then
    echo "[cluv job] no program given -- submit with: cluv submit <cluster> -- <command> [args...]" >&2
    exit 1
fi

# Retry fast startup failures (shared-FS flakiness: triton JIT getsource
# errors, NCCL rendezvous timeouts, HF cache lock contention). A failure
# after >10 min is real.
for ATTEMPT in 1 2 3; do
    START=$(date +%s)
    "$@" && exit 0
    ELAPSED=$(( $(date +%s) - START ))
    if [ $ELAPSED -gt 600 ]; then
        echo "[retry] failure after ${ELAPSED}s, not retrying"
        exit 1
    fi
    echo "[retry] fast startup failure (attempt $ATTEMPT, ${ELAPSED}s), retrying in 60s..."
    sleep 60
done
exit 1
