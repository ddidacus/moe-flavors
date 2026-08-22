#!/bin/bash
# Submit the prompt-conditioned cache-size GRPO job via cluv -- trains
# scripts/train/train_prompt_conditioned.py: pure GRPO/DAPO (no SFT term),
# --beta 0.04, cache size conditioned via a "[CACHE_SIZE=X]" prompt prefix
# (X in {2,4,8}) parsed back out per-example at reward time, instead of
# finetune_moe_grpo.py's --conditioned-cache-sizes embedding mechanism.
#
# Sizing: --max-samples 10000 is the BASE (unprefixed) prompt count; the
# dataset the script actually builds has 10000 x 3 = 30000 rows (one row per
# cache size). prompt-len/completion-len = 1024/1024 and batch-size=16,
# grad-accum=1 mirror scripts/cluv/train_cache_reward.sh's config, which is
# the proven-safe single-A100L/H100 memory profile for the NO-SFT variant at
# this context length (smoke_test_context.sh; cache_reward has no SFT NLL
# forward pass, so it fits at batch=16 where cache_sft needed batch=8 x
# grad-accum=2 to avoid OOM -- this script is architecturally identical to
# cache_reward on that axis, just with a per-example instead of per-step
# cache size). --cache-experts-per-token 2 --cache-topk mirrors the
# deterministic top-2 routing cache_reward/cache_sft use (matches phimoe's
# actual top-2 routing instead of sampling).
#
# num-steps 625 = max-samples(10000) / batch-size(16), the same "max_samples
# = num_steps x batch_size" sizing heuristic used by train_cache_reward.sh/
# train_cache_sft.sh (every step sees an unrepeated per-device batch, no
# epoch-repeat guarantee -- ignores the num_generations/num_gpus grouping
# factor, same simplification those scripts already make). At
# cache_reward's observed ~140s/step (200 steps ~7.8h, --soft-cache dense
# routing -- this run is cheaper, sparse top-2 routing, no dense-router
# patch, so treat 140s/step as a conservative upper bound), 625 steps is
# ~24.3h -- --time gives it 30h of headroom in case per-step cost isn't
# actually lower. --resume is set: if the walltime cap is hit before
# global_step==625, resubmit this same script (CLUSTER=tamia bash
# scripts/cluv/train_prompt_conditioned.sh) to continue from the last
# checkpoint.
#
# Usage: CLUSTER=tamia bash scripts/cluv/train_prompt_conditioned.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-06:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/train/train_prompt_conditioned.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 10000 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --save-total-limit 3 --resume \
    --batch-size 16 --gradient-accumulation-steps 1 --num-steps 625 \
    --num-generations 8 --temperature 1.0 --beta 0.04 \
    --cache-layer -1 --cache-experts-per-token 2 --cache-topk \
    --prompt-cache-sizes 2,4,8 \
    --eval-ppl-every 10 --eval-hitrate-every 10 \
    --wandb-run-name "prompt-conditioned-${CLUSTER}" --save-dir "checkpoints/prompt_conditioned_${CLUSTER}"
echo "prompt_conditioned -> submitted to $CLUSTER"
