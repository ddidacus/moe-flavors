#!/bin/bash
#SBATCH --job-name=eval_soft_cache_1gpu
#SBATCH --output=eval_soft_cache_1gpu_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100l:1
#SBATCH --partition=long
#SBATCH --time=04:00:00

# Single-GPU sequential variant of run_router.sh: runs each variant one
# after another on the one A100 instead of one-GPU-per-variant in parallel.
#
# --out-dir defaults to evals/router_<today>; override with OUT_DIR=...
#
# Per-variant checkpoint override: set CHECKPOINT_DIR_<VARIANT> (uppercase,
# hyphens/dashes as underscores) -- see run_router.sh for details.
#
# Usage: sbatch scripts/eval/run_router_1gpu.sh
#        (defaults to all 6 variants; pass explicit names to override, e.g.
#        sbatch scripts/eval/run_router_1gpu.sh cache_sft temporal_moe)

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache

OUT_DIR="${OUT_DIR:-evals/router_$(date +%F)}"
VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
    VARIANTS=(base sft_baseline cache_sft controller_baseline melinoe temporal_moe)
fi

status=0
for variant in "${VARIANTS[@]}"; do
    ckpt_args=()
    if [ "$variant" != "base" ]; then
        env_name="CHECKPOINT_DIR_$(echo "$variant" | tr '[:lower:]-' '[:upper:]_')"
        ckpt_dir="${!env_name:-}"
        if [ -n "$ckpt_dir" ]; then ckpt_args=(--checkpoint-dir "$ckpt_dir"); fi
    fi
    TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${variant} \
        python scripts/eval/eval_router.py --variant "$variant" "${ckpt_args[@]}" \
        --out-dir "$OUT_DIR" || status=1
done
exit $status
