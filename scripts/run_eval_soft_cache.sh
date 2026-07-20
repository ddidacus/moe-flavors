#!/bin/bash
#SBATCH --job-name=eval_softcache
#SBATCH --output=eval_softcache_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --partition=main
#SBATCH --time=3:00:00

# One job, two GPUs, two processes: base and tuned run concurrently (each
# pinned to its own GPU), then a lightweight CPU-only merge writes
# metrics.json + all plots from the two partial states. Do not pass
# --variant yourself -- it's appended last on each invocation below so it
# always wins over anything in "$@".

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_base \
    python scripts/eval_soft_cache.py "$@" --variant base &
pid_base=$!

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_tuned \
    python scripts/eval_soft_cache.py "$@" --variant tuned &
pid_tuned=$!

wait "$pid_base"; status_base=$?
wait "$pid_tuned"; status_tuned=$?

if [ "$status_base" -ne 0 ] || [ "$status_tuned" -ne 0 ]; then
    echo "[run_eval_soft_cache] a variant process failed (base=$status_base, tuned=$status_tuned)" >&2
    exit 1
fi

python scripts/eval_soft_cache.py "$@" --variant merge
