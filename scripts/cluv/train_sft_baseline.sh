#!/bin/bash
# Submit the sft_baseline small-scale training job via cluv (see
# scripts/train_small_scale.sh for the config this mirrors, and
# scripts/train/run_sft.sh for the mila/sbatch equivalent).
#
# max-samples 3200 = num-steps(200) x batch-size(16), same unified training
# budget as cache_sft/cache_reward/controller_baseline/melinoe (see
# train_cache_sft.sh's comment for the smoke-test rationale) -- this
# replaces the old 32-step/2000-sample config (checkpoints/phi-tiny-moe-sft
# locally) with one directly comparable to the other four variants.
#
# --max-scan-per-split 300000 draws that 3200-row reservoir sample from up
# to 300k rows of EACH of the 9 splits (2.7M rows scanned total) instead of
# the default 50k/split -- covers stem/math/code (355k/239k/175k rows) in
# full and a meaningfully larger, less front-of-stream-biased slice of the
# ~1M-row chat/multilingual splits, without the many-hour streaming+
# tokenization cost of scanning literally every row of all 9 splits
# (~6.3M rows / ~98GB text). Prompts are FILTERED to prompt-len, not
# truncated (see src.nemotron_data.sample_filtered_prompts). SFTConfig's
# max_steps overrides num_train_epochs in HF Trainer, so --num-steps 200
# governs regardless of the epoch count. Saves to
# checkpoints/sft_baseline_<CLUSTER> -- matches what
# scripts/cluv/eval_benchmarks.sh / eval_router.sh look up by default.
#
# Usage: CLUSTER=fir bash scripts/cluv/train_sft_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/train/finetune_moe_sft.py \
    --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
    --dataset-split stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr \
    --max-samples 3200 --max-scan-per-split 300000 --prompt-len 1024 --completion-len 1024 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 --seed 42 \
    --wandb-project moe-cache-reinforce --save-every 50 --resume \
    --batch-size 16 --gradient-accumulation-steps 1 --num-steps 200 \
    --wandb-run-name "sft-baseline-${CLUSTER}" --save-dir "checkpoints/sft_baseline_${CLUSTER}"
echo "sft_baseline -> submitted to $CLUSTER"
