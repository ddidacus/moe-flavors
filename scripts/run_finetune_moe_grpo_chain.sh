#!/bin/bash
#SBATCH --job-name=grpo_chain
#SBATCH --output=grpo_chain_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120

# Runs cache_sft and temporal_moe CONCURRENTLY within one 4-GPU
# short-unkillable allocation -- 2 GPUs each, as two background
# `accelerate launch` processes -- instead of splitting the same 4 GPUs
# into sequential time-boxed slices. Each SLURM submission gets one 3h
# window; --signal=B:USR1@120 + each run's own Preemption handler
# checkpoints before the job is killed at the time limit. This script
# resubmits itself at the end if either run hasn't reached NUM_STEPS yet.
#
# GRAD_ACCUM defaults to 4 here (vs 2 in the 4-GPU standalone scripts) to
# keep the effective global batch size the same: 8 (per-device) x 4
# (accum) x 2 (GPUs) = 64, matching 8 x 2 x 4 = 64 on 4 GPUs -- important
# since cache_sft already has real optimizer steps checkpointed at that
# effective batch size and must resume train with the same dynamics.

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
DATASET_SPLIT="${DATASET_SPLIT:-stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
COMPLETION_LEN="${COMPLETION_LEN:-1024}"
LR="${LR:-1e-4}"
NUM_STEPS="${NUM_STEPS:-250}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
BETA="${BETA:-0.08}"
RL_COEF="${RL_COEF:-2.0}"
SFT_COEF="${SFT_COEF:-0.5}"
CACHE_SIZE="${CACHE_SIZE:-4}"
CACHE_LAYER="${CACHE_LAYER:--1}"
CACHE_EXPERTS="${CACHE_EXPERTS:-2}"
CACHE_TOPK="${CACHE_TOPK:-1}"
RATIO_N="${RATIO_N:-8}"
NUM_GEN="${NUM_GEN:-8}"
SAVE_EVERY="${SAVE_EVERY:-25}"
DATA_TAG=$([ "$DATASET_SPLIT" = "math,code" ] && echo "mathcode" || echo "allsplits")

# Mirrors run_finetune_moe_grpo.sh's SAVE_DIR/RUN_NAME naming exactly, so
# checkpoints land in the same place eval scripts already expect.
compute_names() {
    local REWARD_TAG
    REWARD_TAG=$([ "$CACHE_TOPK" = "1" ] && echo "topk${CACHE_EXPERTS}" || echo "sampled${CACHE_EXPERTS}")
    if [ "$SOFT_CACHE" = "1" ]; then REWARD_TAG="softall"; fi
    if [ "$TEMPORAL" = "1" ]; then REWARD_TAG="${REWARD_TAG}_tmoeN${RATIO_N}"; fi
    if [ "$LR" != "3e-5" ]; then REWARD_TAG="${REWARD_TAG}_lr${LR}"; fi
    if [ "$NUM_STEPS" != "500" ]; then REWARD_TAG="${REWARD_TAG}_dbg${NUM_STEPS}"; fi
    if [ "$RL_COEF" != "1.0" ]; then REWARD_TAG="${REWARD_TAG}_rl${RL_COEF}"; fi
    if [ "$PROMPT_LEN" != "512" ] || [ "$COMPLETION_LEN" != "512" ]; then
        REWARD_TAG="${REWARD_TAG}_seq${PROMPT_LEN}-${COMPLETION_LEN}"
    fi
    if [ "$MAX_SAMPLES" != "20000" ]; then REWARD_TAG="${REWARD_TAG}_n${MAX_SAMPLES}"; fi
    SAVE_DIR="checkpoints/grpo_${MODEL_TAG}_cache_${DATA_TAG}_sft${SFT_COEF}_b${BETA}_c${CACHE_SIZE}_${REWARD_TAG}"
    RUN_NAME="grpo-${MODEL_TAG}-cache-${DATA_TAG}-sft${SFT_COEF}-b${BETA}-c${CACHE_SIZE}-${REWARD_TAG}"
}

current_step() {  # $1 = SAVE_DIR -> prints global_step of its last checkpoint (0 if none)
    python3 -c "
from transformers.trainer_utils import get_last_checkpoint
import json
d = get_last_checkpoint('$1')
print(json.load(open(d + '/trainer_state.json'))['global_step']) if d else print(0)
" 2>/dev/null || echo 0
}

run_bg() {  # $1=CUDA_VISIBLE_DEVICES  $2=PORT  $3=EXTRA_ARGS  $4=LOG_SUFFIX
    local DEVICES="$1" PORT="$2" EXTRA_ARGS="$3" LOG_SUFFIX="$4"
    mkdir -p .wandb_run_ids
    local WANDB_ID_FILE=".wandb_run_ids/$(basename "$SAVE_DIR")"
    local WANDB_RUN_ID_VAL
    if [ -f "$WANDB_ID_FILE" ]; then
        WANDB_RUN_ID_VAL=$(cat "$WANDB_ID_FILE")
    else
        WANDB_RUN_ID_VAL=$(python3 -c "import wandb; print(wandb.util.generate_id())")
        echo "$WANDB_RUN_ID_VAL" > "$WANDB_ID_FILE"
    fi
    echo "[chain] launching $RUN_NAME on GPUs $DEVICES (save-dir: $SAVE_DIR)"
    (
      export CUDA_VISIBLE_DEVICES="$DEVICES"
      export WANDB_RUN_ID="$WANDB_RUN_ID_VAL"
      export WANDB_RESUME=allow
      export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}_${LOG_SUFFIX}"
      accelerate launch \
        --multi_gpu \
        --num_processes 2 \
        --main_process_port "$PORT" \
        scripts/finetune_moe_grpo.py \
        --model "$MODEL" \
        --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
        --dataset-split "$DATASET_SPLIT" \
        --max-samples "$MAX_SAMPLES" \
        --prompt-len "$PROMPT_LEN" \
        --completion-len "$COMPLETION_LEN" \
        --num-generations "$NUM_GEN" \
        --batch-size "$BATCH_SIZE" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --num-steps "$NUM_STEPS" \
        --num-epochs 10 \
        --lr "$LR" \
        --temperature 1.0 \
        --rl-coef "$RL_COEF" \
        --sft-coef "$SFT_COEF" \
        --beta "$BETA" \
        --cache-size "$CACHE_SIZE" \
        --cache-layer "$CACHE_LAYER" \
        --cache-experts-per-token "$CACHE_EXPERTS" \
        --lora-r 16 \
        --lora-alpha 32 \
        --seed 42 \
        --wandb-project moe-cache-reinforce \
        --wandb-run-name "$RUN_NAME" \
        --save-dir "$SAVE_DIR" \
        --save-every "$SAVE_EVERY" \
        --eval-ppl-every 10 \
        --resume \
        $EXTRA_ARGS
    ) > "grpo_chain_${SLURM_JOB_ID}_${LOG_SUFFIX}.out" 2>&1 &
}

PIDS=()

# --- cache_sft on GPUs 0,1 ---
SOFT_CACHE=1 TEMPORAL=0
compute_names
CACHE_SAVE_DIR="$SAVE_DIR"
if [ "$(current_step "$CACHE_SAVE_DIR")" -lt "$NUM_STEPS" ]; then
    run_bg "0,1" 29500 "--cache-topk --soft-cache" "cache_sft"
    PIDS+=($!)
else
    echo "[chain] cache_sft already at/above $NUM_STEPS steps, skipping"
fi

# --- temporal_moe on GPUs 2,3 ---
SOFT_CACHE=0 TEMPORAL=1
compute_names
TEMPORAL_SAVE_DIR="$SAVE_DIR"
if [ "$(current_step "$TEMPORAL_SAVE_DIR")" -lt "$NUM_STEPS" ]; then
    run_bg "2,3" 29501 "--cache-topk --temporal --ratio-loss-N $RATIO_N" "temporal_moe"
    PIDS+=($!)
else
    echo "[chain] temporal_moe already at/above $NUM_STEPS steps, skipping"
fi

for pid in "${PIDS[@]}"; do wait "$pid"; done

CACHE_STEP=$(current_step "$CACHE_SAVE_DIR")
TEMPORAL_STEP=$(current_step "$TEMPORAL_SAVE_DIR")
echo "[chain] progress: cache_sft ${CACHE_STEP}/${NUM_STEPS}, temporal_moe ${TEMPORAL_STEP}/${NUM_STEPS}"

if [ "$CACHE_STEP" -lt "$NUM_STEPS" ] || [ "$TEMPORAL_STEP" -lt "$NUM_STEPS" ]; then
    echo "[chain] not done -- resubmitting"
    sbatch scripts/run_finetune_moe_grpo_chain.sh
else
    echo "[chain] both runs complete"
fi
