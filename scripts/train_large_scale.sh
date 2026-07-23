#!/bin/bash
# Large-scale training run: submits all 4 models (sft_baseline, cache_sft,
# temporal_moe, controller_baseline) with the "large" config from
# handoff/06-training-setup.md -- 10,000 sequences, all 9 Nemotron-v2
# splits, randomly sampled, prompt+completion = 1024+1024, all on 4x
# A100-80GB. lr and batch/grad-accum per model are each model's own
# empirically-best value from prior runs (see handoff/06 for how they were
# chosen). Each job is independent (own save-dir, own wandb run) and can
# run concurrently.
#
# Usage: bash scripts/train_large_scale.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET_SPLIT="stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr"
MAX_SAMPLES=10000
PROMPT_LEN=1024
COMPLETION_LEN=1024
LR=1e-4

echo "[train_large_scale] dataset: $MAX_SAMPLES sequences across all splits, seq_len=${PROMPT_LEN}+${COMPLETION_LEN}"

j1=$(DATASET_SPLIT="$DATASET_SPLIT" MAX_SAMPLES=$MAX_SAMPLES PROMPT_LEN=$PROMPT_LEN COMPLETION_LEN=$COMPLETION_LEN \
     LR=$LR NUM_STEPS=157 BATCH_SIZE=4 GRAD_ACCUM=4 \
     sbatch --parsable scripts/run_finetune_moe_sft.sh)
echo "sft_baseline       -> job $j1 (157 steps, batch=4/accum=4)"

# cache_sft/temporal_moe: ~198.5/~326.0 GPU-hours estimated (~49.6h/~81.5h
# wall-clock on 4 GPUs) -- exceeds short-unkillable's 3h cap, and main's QOS
# caps at 2 GPUs/user (can't fit a 4-GPU job at all), so both go to `long`
# (a100l:4 nodes there, up to 7-day limit) with a generous 4-day budget.
j2=$(DATASET_SPLIT="$DATASET_SPLIT" MAX_SAMPLES=$MAX_SAMPLES PROMPT_LEN=$PROMPT_LEN COMPLETION_LEN=$COMPLETION_LEN \
     LR=$LR NUM_STEPS=1250 BATCH_SIZE=8 GRAD_ACCUM=2 SOFT_CACHE=1 BETA=0.08 RL_COEF=2.0 SFT_COEF=0.5 \
     sbatch --parsable --partition=long --time=4-00:00:00 scripts/run_finetune_moe_grpo.sh)
echo "cache_sft          -> job $j2 (1250 steps, batch=8/accum=2, long partition, ~2.1d est.)"

j3=$(DATASET_SPLIT="$DATASET_SPLIT" MAX_SAMPLES=$MAX_SAMPLES PROMPT_LEN=$PROMPT_LEN COMPLETION_LEN=$COMPLETION_LEN \
     LR=$LR NUM_STEPS=1250 BATCH_SIZE=8 GRAD_ACCUM=2 TEMPORAL=1 CACHE_TOPK=1 BETA=0.08 RL_COEF=2.0 SFT_COEF=0.5 \
     sbatch --parsable --partition=long --time=4-00:00:00 scripts/run_finetune_moe_grpo.sh)
echo "temporal_moe       -> job $j3 (1250 steps, batch=8/accum=2, long partition, ~3.4d est.)"

j4=$(DATASET_SPLIT="$DATASET_SPLIT" MAX_SAMPLES=$MAX_SAMPLES PROMPT_LEN=$PROMPT_LEN COMPLETION_LEN=$COMPLETION_LEN \
     LR=$LR NUM_STEPS=157 BATCH_SIZE=4 GRAD_ACCUM=4 \
     sbatch --parsable scripts/run_finetune_moe_controller.sh)
echo "controller_baseline -> job $j4 (157 steps, batch=4/accum=4)"

echo "[train_large_scale] all 4 jobs submitted: $j1 $j2 $j3 $j4"
