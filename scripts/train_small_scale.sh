#!/bin/bash
# Small-scale training run: submits all 4 models (sft_baseline, cache_sft,
# temporal_moe, controller_baseline) with the "small" config from
# handoff/08-training-setup-small.md -- 2,000 sequences (closer to the
# debug-scale runs already completed in past days), all 9 Nemotron-v2
# splits, randomly sampled, prompt+completion = 1024+1024, all on 4x
# A100-80GB. Same lr/batch/grad-accum per model as the large-scale config
# (scripts/train_large_scale.sh) -- only the dataset size and step count
# change.
#
# Usage: bash scripts/train_small_scale.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET_SPLIT="stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr"
MAX_SAMPLES=2000
PROMPT_LEN=1024
COMPLETION_LEN=1024
LR=1e-4

echo "[train_small_scale] dataset: $MAX_SAMPLES sequences across all splits, seq_len=${PROMPT_LEN}+${COMPLETION_LEN}"

j1=$(DATASET_SPLIT="$DATASET_SPLIT" MAX_SAMPLES=$MAX_SAMPLES PROMPT_LEN=$PROMPT_LEN COMPLETION_LEN=$COMPLETION_LEN \
     LR=$LR NUM_STEPS=32 BATCH_SIZE=4 GRAD_ACCUM=4 \
     sbatch --parsable scripts/run_finetune_moe_sft.sh)
echo "sft_baseline       -> job $j1 (32 steps, batch=4/accum=4)"

# cache_sft/temporal_moe: ~39.7/~65.2 GPU-hours estimated (~9.9h/~16.3h
# wall-clock on 4 GPUs) -- still exceeds short-unkillable's 3h cap, and
# main's QOS caps at 2 GPUs/user (can't fit a 4-GPU job at all), so both go
# to `long` (a100l:4 nodes there) with a 1-day budget.
j2=$(DATASET_SPLIT="$DATASET_SPLIT" MAX_SAMPLES=$MAX_SAMPLES PROMPT_LEN=$PROMPT_LEN COMPLETION_LEN=$COMPLETION_LEN \
     LR=$LR NUM_STEPS=250 BATCH_SIZE=8 GRAD_ACCUM=2 SOFT_CACHE=1 BETA=0.08 RL_COEF=2.0 SFT_COEF=0.5 \
     sbatch --parsable --partition=long --time=1-00:00:00 scripts/run_finetune_moe_grpo.sh)
echo "cache_sft          -> job $j2 (250 steps, batch=8/accum=2, long partition, ~9.9h est.)"

j3=$(DATASET_SPLIT="$DATASET_SPLIT" MAX_SAMPLES=$MAX_SAMPLES PROMPT_LEN=$PROMPT_LEN COMPLETION_LEN=$COMPLETION_LEN \
     LR=$LR NUM_STEPS=250 BATCH_SIZE=8 GRAD_ACCUM=2 TEMPORAL=1 CACHE_TOPK=1 BETA=0.08 RL_COEF=2.0 SFT_COEF=0.5 \
     sbatch --parsable --partition=long --time=1-00:00:00 scripts/run_finetune_moe_grpo.sh)
echo "temporal_moe       -> job $j3 (250 steps, batch=8/accum=2, long partition, ~16.3h est.)"

j4=$(DATASET_SPLIT="$DATASET_SPLIT" MAX_SAMPLES=$MAX_SAMPLES PROMPT_LEN=$PROMPT_LEN COMPLETION_LEN=$COMPLETION_LEN \
     LR=$LR NUM_STEPS=32 BATCH_SIZE=4 GRAD_ACCUM=4 \
     sbatch --parsable scripts/run_finetune_moe_controller.sh)
echo "controller_baseline -> job $j4 (32 steps, batch=4/accum=4)"

echo "[train_small_scale] all 4 jobs submitted: $j1 $j2 $j3 $j4"
