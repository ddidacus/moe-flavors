#!/bin/bash
#SBATCH --job-name=eval_soft_cache
#SBATCH --output=eval_soft_cache_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=long
#SBATCH --time=1-00:00:00

# Takes one or more variant names as positional args, each pinned to its own
# GPU (CUDA_VISIBLE_DEVICES=0,1,...) and run in parallel as background
# processes within this single job -- same pattern as run_eval_lm_harness.sh.
# Pass up to 4 variants. On `long` (not short-unkillable): this eval is
# on-policy (generates a completion per held-out prompt at T=1.0, then
# scores cache-hit rate on the generated tokens only -- see
# eval_soft_cache.py's docstring), which is much slower than the old
# teacher-forced version and doesn't reliably fit short-unkillable's 3h cap.
# --out-dir defaults to evals/soft_cache_<today>; override with OUT_DIR=...
# if needed.
#
# Per-variant checkpoint override: set CHECKPOINT_DIR_<VARIANT> (uppercase,
# hyphens/dashes as underscores) to point at a checkpoint trained elsewhere
# (e.g. a small-scale run synced from cluv) instead of
# eval_lm_harness.py's VARIANT_CHECKPOINTS[variant] (the mila
# run_finetune_moe_*.sh naming convention). "base" never takes a checkpoint.
#
# Usage: sbatch scripts/run_eval_soft_cache.sh cache_sft temporal_moe
#        CHECKPOINT_DIR_CACHE_SFT=checkpoints/small-scale/cache_sft_tamia \
#            sbatch scripts/run_eval_soft_cache.sh cache_sft

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache

OUT_DIR="${OUT_DIR:-evals/soft_cache_$(date +%F)}"

pids=()
gpu=0
for variant in "$@"; do
    ckpt_args=()
    if [ "$variant" != "base" ]; then
        env_name="CHECKPOINT_DIR_$(echo "$variant" | tr '[:lower:]-' '[:upper:]_')"
        ckpt_dir="${!env_name:-}"
        if [ -n "$ckpt_dir" ]; then ckpt_args=(--checkpoint-dir "$ckpt_dir"); fi
    fi
    CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${variant} \
        python scripts/eval_soft_cache.py --variant "$variant" "${ckpt_args[@]}" \
        --out-dir "$OUT_DIR" &
    pids+=($!)
    gpu=$((gpu + 1))
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit $status
