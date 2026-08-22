#!/bin/bash
# Run scripts/eval/eval_cache_conditioning.py on mila, single GPU (no NCCL/
# DDP needed -- this is inference-only, no distributed training). Compares
# the prompt-conditioned GRPO checkpoint (checkpoints/prompt_conditioned_
# test_mila, job 10400333) against the untouched base model, for each
# "[CACHE_SIZE=X]" condition in 2,4,8, on the same 256 held-out prompts.
#
# DRY-RUN mode (default): --num-eval-prompts 16, to confirm both models
# load, generation runs, and both plots render before committing to the
# full 256-prompt run (2 models x 3 cache sizes x 256 prompts x up to 1024
# generated tokens each, on-policy -- will take a while). Set
# NUM_EVAL_PROMPTS=256 to run the real thing.
#
# No wandb involved here (this is a standalone eval script, no GRPOConfig/
# Trainer) -- results go to evals/cache_conditioning/results.json + PNGs
# and the SLURM stdout log only, so no DEV_MODE/--export needed.
#
# results.json is written incrementally (atomic per-combo replace), but
# only within ONE process -- running this alongside another eval_cache_
# conditioning*.sh job pointed at the SAME --out-dir means whichever job
# finishes a combo last wins the shared file (each job's own SLURM log is
# still the authoritative record either way). scripts/cluv/eval_cache_
# conditioning_long.sh (the `long`/l40s variant) defaults to a different
# --out-dir for exactly this reason; override OUT_DIR here too if running
# both concurrently.
#
# --out-dir defaults to evals/cache_conditioning_dryrun for the default
# (dry-run) --num-eval-prompts, and only evals/cache_conditioning (the
# "real"/final results dir) when NUM_EVAL_PROMPTS is explicitly overridden
# -- a dry run submitted with the plain default OUT_DIR once clobbered a
# completed 256-prompt run's results.json mid-write (had to restore from a
# manual backup). Set OUT_DIR explicitly to override either default.
#
# Usage: bash scripts/cluv/eval_cache_conditioning.sh
#        NUM_EVAL_PROMPTS=256 bash scripts/cluv/eval_cache_conditioning.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

NUM_EVAL_PROMPTS="${NUM_EVAL_PROMPTS:-16}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/prompt_conditioned_test_mila}"
if [ "$NUM_EVAL_PROMPTS" = "16" ]; then
    OUT_DIR="${OUT_DIR:-evals/cache_conditioning_dryrun}"
else
    OUT_DIR="${OUT_DIR:-evals/cache_conditioning}"
fi

cluv submit --autocommit mila --partition=short-unkillable --time=3:00:00 -- env CUDA_VISIBLE_DEVICES=0 \
    python scripts/eval/eval_cache_conditioning.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-sizes 2,4,8 --cache-layer -1 --cache-experts-per-token 2 --cache-topk \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 --dataset-split math,code \
    --num-eval-prompts "$NUM_EVAL_PROMPTS" --prompt-len 1024 --gen-len 1024 \
    --seed 42 --batch-size 16 --out-dir "$OUT_DIR"
echo "eval_cache_conditioning -> submitted to mila (num_eval_prompts=$NUM_EVAL_PROMPTS, out_dir=$OUT_DIR)"
