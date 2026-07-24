#!/bin/bash
# Small-scale training run: submits all 4 models (sft_baseline, cache_sft,
# temporal_moe, controller_baseline) with the "small" config from
# handoff/08-training-setup-small.md -- 2,000 sequences (closer to the
# debug-scale runs already completed in past days), all 9 Nemotron-v2
# splits, randomly sampled, prompt+completion = 1024+1024, all on 4 GPUs.
# Same lr/batch/grad-accum per model as the large-scale config
# (scripts/train_large_scale.sh) -- only the dataset size and step count
# change.
#
# Usage: bash scripts/train_small_scale.sh
#        CLUSTER=fir bash scripts/train_small_scale.sh   # submit via cluv instead of sbatch
#
# CLUSTER defaults to "mila", which submits directly with `sbatch` against
# the existing scripts/run_finetune_moe_*.sh job scripts (unchanged
# behavior). Any other CLUSTER value (tamia, rorqual, narval, vulcan, fir,
# nibi -- see scripts/cluv/README.md) submits the same 4 jobs through
# `cluv submit`, using that cluster's scripts/cluv/<cluster>_job.sh
# (wired up via job_script_path in pyproject.toml) for the GPU/node request.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER="${CLUSTER:-mila}"

DATASET_SPLIT="stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr"
MAX_SAMPLES=2000
PROMPT_LEN=1024
COMPLETION_LEN=1024
LR=1e-4

echo "[train_small_scale] cluster: $CLUSTER, dataset: $MAX_SAMPLES sequences across all splits, seq_len=${PROMPT_LEN}+${COMPLETION_LEN}"

if [ "$CLUSTER" = "mila" ]; then
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
    exit 0
fi

# --- non-mila cluster: submit via cluv, program args after `--' ---
COMMON_ARGS=(--dataset nvidia/Nemotron-Post-Training-Dataset-v2
             --dataset-split "$DATASET_SPLIT" --max-samples "$MAX_SAMPLES"
             --prompt-len "$PROMPT_LEN" --completion-len "$COMPLETION_LEN"
             --lr "$LR" --lora-r 16 --lora-alpha 32 --seed 42
             --wandb-project moe-cache-reinforce --save-every 50 --resume)

cluv submit --autocommit "$CLUSTER" -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/finetune_moe_sft.py "${COMMON_ARGS[@]}" \
    --batch-size 4 --gradient-accumulation-steps 4 --num-steps 32 --num-epochs 10 \
    --wandb-run-name "sft-baseline-${CLUSTER}" --save-dir "checkpoints/sft_baseline_${CLUSTER}"
echo "sft_baseline        -> submitted to $CLUSTER (32 steps, batch=4/accum=4)"

# cache_sft/temporal_moe: ~9.9h/~16.3h wall-clock on 4 GPUs estimated (see
# mila branch above) -- override walltime past the 3h pyproject.toml default.
cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/finetune_moe_grpo.py "${COMMON_ARGS[@]}" \
    --batch-size 8 --gradient-accumulation-steps 2 --num-steps 250 --num-epochs 10 \
    --num-generations 8 --temperature 1.0 --rl-coef 2.0 --sft-coef 0.5 --beta 0.08 \
    --cache-size 4 --cache-layer -1 --cache-experts-per-token 2 --cache-topk --soft-cache \
    --eval-ppl-every 10 \
    --wandb-run-name "cache-sft-${CLUSTER}" --save-dir "checkpoints/cache_sft_${CLUSTER}"
echo "cache_sft           -> submitted to $CLUSTER (250 steps, batch=8/accum=2, ~9.9h est.)"

cluv submit --autocommit "$CLUSTER" --time=1-00:00:00 -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/finetune_moe_grpo.py "${COMMON_ARGS[@]}" \
    --batch-size 8 --gradient-accumulation-steps 2 --num-steps 250 --num-epochs 10 \
    --num-generations 8 --temperature 1.0 --rl-coef 2.0 --sft-coef 0.5 --beta 0.08 \
    --cache-size 4 --cache-layer -1 --cache-experts-per-token 2 --cache-topk \
    --temporal --ratio-loss-N 8 --eval-ppl-every 10 \
    --wandb-run-name "temporal-moe-${CLUSTER}" --save-dir "checkpoints/temporal_moe_${CLUSTER}"
echo "temporal_moe        -> submitted to $CLUSTER (250 steps, batch=8/accum=2, ~16.3h est.)"

cluv submit --autocommit "$CLUSTER" -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/finetune_moe_controller.py "${COMMON_ARGS[@]}" \
    --batch-size 4 --gradient-accumulation-steps 4 --num-steps 32 \
    --cache-size 4 --cache-layer -1 --deliberation-cost 0.02 \
    --wandb-run-name "controller-baseline-${CLUSTER}" --save-dir "checkpoints/controller_baseline_${CLUSTER}"
echo "controller_baseline -> submitted to $CLUSTER (32 steps, batch=4/accum=4)"

echo "[train_small_scale] all 4 jobs submitted to $CLUSTER via cluv"
