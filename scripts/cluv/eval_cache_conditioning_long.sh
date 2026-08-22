#!/bin/bash
# Same as scripts/cluv/eval_cache_conditioning.sh, but on mila's `long`
# partition instead of short-unkillable -- avoids the 3h wall-clock cap
# entirely (a prior short-unkillable attempt, job 10403943, needed ~3h for
# all 9 (model, cache_size) combos and got killed at 2h57m with 2 combos
# left; see scripts/eval/eval_cache_conditioning.py's incremental results.
# json write for the safety net either way). This is inference-only,
# single-GPU, so a small allocation is enough: --gres=gpu:l40s:1
# --cpus-per-task=4 --mem=16G (vs. job.sh's hardcoded --gres=gpu:a100l:4
# --cpus-per-task=24 --mem=200G default for the whole-node short-unkillable
# jobs elsewhere in this directory -- these flags on the `cluv submit`
# command line override job.sh's #SBATCH header, same mechanism already
# used for --partition on the short-unkillable variant).
#
# `long` allows much longer walltime than short-unkillable's 3h; --time
# below is a generous cap, not an expected duration.
#
# No explicit CUDA_VISIBLE_DEVICES here (unlike the short-unkillable
# variant, which pins CUDA_VISIBLE_DEVICES=0 on a whole-node 4-GPU
# allocation to select just one): with --gres=gpu:l40s:1, SLURM's cgroup
# already restricts the job to exactly the granted GPU and sets
# CUDA_VISIBLE_DEVICES accordingly. Forcing =0 on top of that collided
# with the cgroup's own device mapping and crashed every attempt with
# "CUDA driver initialization failed, you might not have a CUDA gpu" even
# though nvidia-smi showed the L40S right there (job 10410881) -- same
# pattern as scripts/eval/run_router_1gpu.sh, which also doesn't set it.
#
# --out-dir defaults to evals/cache_conditioning_long, distinct from the
# short-unkillable variant's evals/cache_conditioning -- both scripts write
# results.json incrementally (atomic per-combo replace, see the script),
# but that atomicity is only per-process: if two *separate* jobs point at
# the SAME out-dir (as an earlier pair of runs briefly did, 10410868 +
# 10411478/10410881), whichever finishes a combo last wins the shared
# file, so the on-disk state can appear to "regress" to an earlier job's
# progress even though nothing is actually lost (each job's own combo
# lines in its SLURM log are still the authoritative record). Separate
# --out-dirs avoid that confusion entirely for future concurrent runs.
#
# Additionally, a dry run (default --num-eval-prompts 16) uses its own
# _dryrun suffix rather than the plain evals/cache_conditioning_long dir --
# a dry run of the OTHER (short-unkillable) variant once clobbered a
# completed 256-prompt run's results.json mid-write because both dry-run
# and full-run shared one default out-dir. Set OUT_DIR explicitly to
# override either default.
#
# Usage: bash scripts/cluv/eval_cache_conditioning_long.sh
#        NUM_EVAL_PROMPTS=256 bash scripts/cluv/eval_cache_conditioning_long.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

NUM_EVAL_PROMPTS="${NUM_EVAL_PROMPTS:-16}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/prompt_conditioned_test_mila}"
if [ "$NUM_EVAL_PROMPTS" = "16" ]; then
    OUT_DIR="${OUT_DIR:-evals/cache_conditioning_long_dryrun}"
else
    OUT_DIR="${OUT_DIR:-evals/cache_conditioning_long}"
fi

cluv submit --autocommit mila --partition=long --gres=gpu:l40s:1 \
    --cpus-per-task=4 --mem=16G --time=08:00:00 -- \
    python scripts/eval/eval_cache_conditioning.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-sizes 2,4,8 --cache-layer -1 --cache-experts-per-token 2 --cache-topk \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 --dataset-split math,code \
    --num-eval-prompts "$NUM_EVAL_PROMPTS" --prompt-len 1024 --gen-len 1024 \
    --seed 42 --batch-size 16 --out-dir "$OUT_DIR"
echo "eval_cache_conditioning (long, l40s:1) -> submitted to mila (num_eval_prompts=$NUM_EVAL_PROMPTS, out_dir=$OUT_DIR)"
