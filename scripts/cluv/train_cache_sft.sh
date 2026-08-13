#!/bin/bash
# Submit the cache_sft small-scale training job via cluv (see
# scripts/train_small_scale.sh for the config this mirrors, and
# scripts/train/run_grpo.sh for the mila/sbatch equivalent).
#
# prompt-len/completion-len=1024/1024, batch-size=16, no gradient
# accumulation: scripts/train/smoke_test_context.sh confirmed 2048+2048
# OOMs a single 80GB A100L at batch=16 (peak ~78GB before the crash), while
# 1024+1024 fits -- barely (peak ~78.6/80GB on the smoke test's 3-step
# probe, so there's very little headroom; if this OOMs partway through a
# real run, --gradient-accumulation-steps 2 --batch-size 8 is the fallback
# that keeps the same effective batch). Prompts are FILTERED to prompt-len,
# not truncated (see src.nemotron_data.sample_filtered_prompts) -- rows
# with a too-long prompt are dropped from the sample instead. max-samples
# 3200 = num-steps(200) x batch-size(16), so every step sees a fresh,
# unrepeated sample given no epoch repeats within a single run.
# ~7.8h wall-clock estimated on 4 GPUs -- overrides the 3h pyproject.toml
# default walltime. Saves to checkpoints/cache_sft_<CLUSTER>_v2 -- the "_v2"
# avoids --resume picking up the OLD checkpoints/cache_sft_<CLUSTER> run
# (already at global_step==max_steps under the old truncation-based data
# pipeline and hyperparameters), which would otherwise silently no-op
# instead of actually retraining under the new filtering/config.
#
# Usage: CLUSTER=fir bash scripts/cluv/train_cache_sft.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/train/finetune_moe_grpo.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 3200 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --save-total-limit 3 --resume \
    --batch-size 16 --gradient-accumulation-steps 1 --num-steps 200 \
    --num-generations 8 --temperature 1.0 --rl-coef 2.0 --sft-coef 0.5 --beta 0.08 \
    --cache-size 4 --cache-layer -1 --cache-experts-per-token 2 --cache-topk --soft-cache \
    --eval-ppl-every 10 \
    --wandb-run-name "cache-sft-${CLUSTER}-v2" --save-dir "checkpoints/cache_sft_${CLUSTER}_v2"
echo "cache_sft -> submitted to $CLUSTER"
