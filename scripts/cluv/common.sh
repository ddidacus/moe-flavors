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

# UV_OFFLINE=1 (set per-cluster in pyproject.toml's [tool.cluv.env]) already
# marks "this cluster has no internet" -- reuse that same signal here.
# Without this, `from_pretrained("microsoft/Phi-tiny-MoE-instruct")` retries
# HEAD requests to huggingface.co for ~10+ min before giving up (not gated,
# but still needs a network round-trip to resolve). datasets_path/hf_cache
# is a synced mirror of $HF_HOME/hub for that one model (see pyproject.toml
# [tool.cluv] comment) -- HF_HUB_OFFLINE=1 makes from_pretrained resolve
# from it with zero network calls.
if [ "${UV_OFFLINE:-0}" = "1" ] && [ -d "${SCRATCH:-}/datasets/hf_cache" ]; then
    export HF_HOME="${SCRATCH}/datasets/hf_cache"
    export HF_HUB_OFFLINE=1
fi

# scripts/finetune_moe_*.py write to a plain relative `checkpoints/` dir,
# which otherwise lands on $HOME -- a tiny (e.g. 25GB on tamia) quota
# shared with the venv, vs. $SCRATCH's ~TB-scale quota. A long GRPO run's
# rotating LoRA checkpoints (up to save-total-limit x several GB) can fill
# $HOME outright (EDQUOT mid-write -> corrupted checkpoint, see the
# 2026-07-30 tamia incident). Symlink checkpoints/ to $SCRATCH once,
# migrating any pre-existing on-$HOME checkpoints the first time.
if [ -n "${SCRATCH:-}" ]; then
    mkdir -p "$SCRATCH/checkpoints"
    if [ -d checkpoints ] && [ ! -L checkpoints ]; then
        shopt -s dotglob nullglob
        mv checkpoints/* "$SCRATCH/checkpoints/" 2>/dev/null || true
        rmdir checkpoints 2>/dev/null || true
    fi
    ln -sfn "$SCRATCH/checkpoints" checkpoints
fi

# Compute nodes on these clusters have no internet access, so wandb (even
# in "offline" mode) is more trouble than it's worth -- force it off and
# rely on the SLURM stdout log (%x_%j.out) instead. Overrides any
# WANDB_MODE set via [tool.cluv.env] / [tool.cluv.clusters.*].env.
#
# DEV_MODE=1 opts back into wandb for short interactive test/debug runs
# (e.g. scripts/cluv/train_prompt_conditioned_test.sh), gated to mila only:
# it's the one cluster sharing this job script whose compute nodes actually
# do have internet (confirmed by successful HF Hub requests during job
# 10400333) -- killarney/trillium/trillium-gpu stay force-disabled
# regardless of DEV_MODE, since they're genuinely offline. Set it by
# passing `--export=ALL,DEV_MODE=1` as an sbatch arg to `cluv submit`
# (plain `DEV_MODE=1 cluv submit ...` does NOT propagate -- cluv only
# forwards the specific vars it knows about via pyproject.toml's
# [tool.cluv.env] sections, not the local submitting shell's environment).
if [ "${DEV_MODE:-0}" = "1" ] && [[ "$(hostname -f 2>/dev/null || hostname)" == *mila* ]]; then
    export WANDB_MODE=online
    echo "[cluv job] DEV_MODE=1 on mila -- wandb re-enabled (online)"
else
    export WANDB_MODE=disabled
fi

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
