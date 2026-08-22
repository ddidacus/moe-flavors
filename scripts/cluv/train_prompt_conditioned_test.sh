#!/bin/bash
# TEST RUN of scripts/train/train_prompt_conditioned.py on mila's
# short-unkillable partition (3h wall-clock cap, whole-node 4x A100L) --
# small scale, just to confirm the prompt-conditioned pipeline (dataset
# augmentation, per-example cache-size parsing, GRPO/DAPO-only loss,
# cache_hit_rate_size{2,4,8} logging) runs end to end before the full
# 10k-sample job (scripts/cluv/train_prompt_conditioned.sh) goes out to a
# real allocation (tamia login is currently unreachable).
#
# --max-samples 1000 (-> 3000 dataset rows after x3 cache-size augmentation)
# --num-steps 60 = max-samples(1000)/batch-size(16), same sizing heuristic
# as the full run; short-unkillable's 3h cap only needs this to survive a
# few dozen steps, not finish -- --resume is NOT expected to matter here,
# this run is disposable (checkpoints/prompt_conditioned_test_mila).
#
# --soft-cache added after the first two attempts (jobs 10395292, 10395463)
# both hung identically at step 0: 3 ranks stuck >10min in the post-
# generation completions all_gather (_generate_and_score_completions),
# 1 rank escaping into the next forward pass alone -- same NCCL SeqNum=42
# both times on different nodes, so a deterministic bug, not a flaky node.
# --soft-cache did NOT fix it either (job 10395552, same SeqNum=42, same
# 3-vs-1 split) -- rules out sparse/topk routing as the cause.
#
# --prompt-len/--completion-len cut 1024 -> 256 (job 10395652): also hung
# identically at the same SeqNum=42 despite generation now taking seconds,
# not minutes -- rules out generation speed as the cause.
#
# --num_processes 4 -> 2 (job 10395758): actually completed step 1
# correctly (135.99s/it, matching the proven tamia precedent's ~124-128s/
# step -- real evidence the code itself is correct), then hung on step 2
# with the same NCCL signature. So this isn't purely a 4-rank thing either
# -- points to intermittent NCCL/network flakiness on mila's short-
# unkillable nodes under sustained multi-GPU communication, worse at
# higher rank counts (4 ranks hung immediately, 2 ranks bought one extra
# step) rather than a deterministic bug.
#
# --num_processes 2 -> 1 (single GPU, no NCCL/DDP at all): eliminates the
# whole class of distributed-communication issue above and tests pure
# code correctness in isolation -- no --multi_gpu flag, so accelerate runs
# single-process. If this completes multiple steps cleanly with real
# reward/loss/cache_hit_rate_size{2,4,8} metrics, the code is validated
# and the earlier hangs are conclusively an infra/NCCL issue specific to
# multi-GPU sync on these nodes, not a bug in train_prompt_conditioned.py.
# Confirmed clean on job 10400333: real cache_hit_rate_size2/4/8 values
# logged (0.36/0.55/0.80 at step 10, monotonic with cache size as
# expected), ~125s/step, no NCCL errors.
#
# --export=ALL,DEV_MODE=1: re-enables wandb for this run (see common.sh --
# mila's compute nodes have internet, unlike killarney/trillium/trillium-
# gpu which share the same job script and stay force-disabled regardless).
# Dev/test runs only; the production job (train_prompt_conditioned.sh)
# doesn't set this and logs to the SLURM stdout log only on mila.
#
# Usage: bash scripts/cluv/train_prompt_conditioned_test.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

cluv submit mila --partition=short-unkillable --time=3:00:00 --export=ALL,DEV_MODE=1 -- env CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
    scripts/train/train_prompt_conditioned.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split math,code \
    --max-samples 1000 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 20 --save-total-limit 2 \
    --batch-size 16 --gradient-accumulation-steps 1 --num-steps 60 \
    --num-generations 8 --temperature 1.0 --beta 0.04 \
    --cache-layer -1 --cache-experts-per-token 2 --cache-topk --soft-cache \
    --prompt-cache-sizes 2,4,8 \
    --eval-ppl-every 10 --eval-hitrate-every 10 \
    --wandb-run-name "prompt-conditioned-test-mila" --save-dir "checkpoints/prompt_conditioned_test_mila"
echo "prompt_conditioned test -> submitted to mila (short-unkillable, 3h)"
