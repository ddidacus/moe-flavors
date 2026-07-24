#!/bin/bash
#SBATCH --job-name=grpo_olmoe_cache
#SBATCH --output=grpo_olmoe_cache_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

# Pin the wandb run id once (persisted to a file keyed by SLURM job ID, on
# the shared filesystem -- NOT /tmp, since --requeue can land the same job
# ID on a different node with its own local /tmp) so a preemption +
# automatic requeue continues the same wandb run instead of starting a new
# one. WANDB_RESUME=allow lets wandb attach to an existing run id or create
# it if this is the first attempt.
mkdir -p .wandb_run_ids
WANDB_ID_FILE=".wandb_run_ids/${SLURM_JOB_ID}"
if [ -f "$WANDB_ID_FILE" ]; then
    export WANDB_RUN_ID=$(cat "$WANDB_ID_FILE")
else
    export WANDB_RUN_ID=$(python3 -c "import wandb; print(wandb.util.generate_id())")
    echo "$WANDB_RUN_ID" > "$WANDB_ID_FILE"
fi
export WANDB_RESUME=allow

# --- Configuration ---
MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
RL_COEF="${RL_COEF:-1.0}"    # weight of the GRPO/DAPO policy loss
SFT_COEF="${SFT_COEF:-1.0}"  # weight of the SFT NLL loss added to the policy loss (0 disables)
BETA="${BETA:-0.04}"         # TRL KL(policy||frozen base) coefficient
TEMPORAL="${TEMPORAL:-0}"    # 1: boundary-prediction + hold/switch routing mixin
RATIO_N="${RATIO_N:-8}"      # temporal target segment length
CACHE_SIZE="${CACHE_SIZE:-4}"    # of 16 experts (Phi-tiny); OLMoE: use 16 of 64
CACHE_LAYER="${CACHE_LAYER:--1}"   # -1 = middle layer
NUM_GEN="${NUM_GEN:-8}"      # G: group size (completions per prompt)
LR="${LR:-3e-5}"             # learning rate (non-default gets its own save dir)
CACHE_TOPK="${CACHE_TOPK:-1}"    # 1: deterministic top-k expert reward (deployment-faithful)
CACHE_EXPERTS="${CACHE_EXPERTS:-2}"  # experts scored per token (= router top-k)
SOFT_CACHE="${SOFT_CACHE:-0}"    # 1: dense router at cache layer + soft (weighted) hit reward
NUM_STEPS="${NUM_STEPS:-500}"    # override for short debug runs
BATCH_SIZE="${BATCH_SIZE:-16}"   # per-device batch; lower for --temporal (no grad checkpointing -> more activation mem)
GRAD_ACCUM="${GRAD_ACCUM:-1}"    # raise to compensate when BATCH_SIZE is lowered
PROMPT_LEN="${PROMPT_LEN:-512}"
COMPLETION_LEN="${COMPLETION_LEN:-512}"
DATASET_SPLIT="${DATASET_SPLIT:-math,code}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"

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
DATA_TAG=$([ "$DATASET_SPLIT" = "math,code" ] && echo "mathcode" || echo "allsplits")
SAVE_DIR="checkpoints/grpo_${MODEL_TAG}_cache_${DATA_TAG}_sft${SFT_COEF}_b${BETA}_c${CACHE_SIZE}_${REWARD_TAG}"
RUN_NAME="grpo-${MODEL_TAG}-cache-${DATA_TAG}-sft${SFT_COEF}-b${BETA}-c${CACHE_SIZE}-${REWARD_TAG}"
EXTRA_ARGS=$([ "$CACHE_TOPK" = "1" ] && echo "--cache-topk")
if [ "$SOFT_CACHE" = "1" ]; then EXTRA_ARGS="$EXTRA_ARGS --soft-cache"; fi
if [ "$TEMPORAL" = "1" ]; then EXTRA_ARGS="$EXTRA_ARGS --temporal --ratio-loss-N $RATIO_N"; fi

# Retry fast startup failures (shared-FS flakiness: triton JIT getsource
# errors, HF cache lock contention). A failure after >10 min is real.
for ATTEMPT in 1 2 3; do
    START=$(date +%s)
    accelerate launch \
    --multi_gpu \
    --num_processes 4 \
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
    --save-every 50 \
    --eval-ppl-every 10 \
    --resume \
    $EXTRA_ARGS && break
    ELAPSED=$(( $(date +%s) - START ))
    if [ $ELAPSED -gt 600 ]; then echo "[retry] failure after ${ELAPSED}s, not retrying"; break; fi
    echo "[retry] fast startup failure (attempt $ATTEMPT, ${ELAPSED}s), retrying in 60s..."
    sleep 60
done
[ "$ATTEMPT" = "3" ] && [ ! -d "$SAVE_DIR" ] && { echo "[retry] all attempts failed, no checkpoint saved"; exit 1; }
exit 0
