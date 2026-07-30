#!/bin/bash
#SBATCH --job-name=grpo_tmoe_sftinit_chain
#SBATCH --output=grpo_tmoe_sftinit_chain_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120

# Temporal-MoE variant of run_finetune_moe_sft_then_dapo_chain.sh: (1) SFT
# with LoRA on the TemporalWrapMixin-wrapped model (see
# run_finetune_moe_sft.sh --temporal, a new flag added alongside this
# script), (2) DAPO RL (cache-hit reward, --temporal, NOT --soft-cache --
# incompatible with the hold/switch mixin) initialized from that SFT
# adapter's weights via --init-adapter. Unlike the softall sft_then_dapo
# variant, SFT_COEF=0 here: no NLL loss and no distillation/KD reward term
# added to the DAPO objective, just the raw cache-hit-ratio reward -- pure
# RL fine-tuning on top of a temporal-SFT'd starting point. BETA stays at
# 0.08 (matching the from-scratch temporal_moe run), not the raised 0.1
# used for the non-temporal sft_then_dapo variant (that raise was
# specifically to protect plain-SFT quality against RL drift towards a
# term this variant doesn't have anyway).
#
# Same chaining mechanics as run_finetune_moe_sft_then_dapo_chain.sh:
# timeout 10200 (< 3h SLURM cap) so the wrapper always regains control to
# run its own resubmit logic, plus a 3-attempt retry loop for transient
# shared-FS/triton-JIT startup failures.

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
DATASET_SPLIT="${DATASET_SPLIT:-stem,chat,math,code,multilingual_ja,multilingual_de,multilingual_it,multilingual_es,multilingual_fr}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
COMPLETION_LEN="${COMPLETION_LEN:-1024}"
LR="${LR:-1e-4}"
NUM_STEPS="${NUM_STEPS:-250}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
BETA="${BETA:-0.08}"
RL_COEF="${RL_COEF:-1.0}"
SFT_COEF="${SFT_COEF:-0.0}"
CACHE_SIZE="${CACHE_SIZE:-4}"
CACHE_LAYER="${CACHE_LAYER:--1}"
CACHE_EXPERTS="${CACHE_EXPERTS:-2}"
RATIO_N="${RATIO_N:-8}"
SAVE_EVERY="${SAVE_EVERY:-10}"
INIT_ADAPTER="${INIT_ADAPTER:-checkpoints/sft_phi-tiny-moe-instruct_allsplits_lr1e-4_seq1024-1024_n2000_tmoe/checkpoint-32}"
DATA_TAG=$([ "$DATASET_SPLIT" = "math,code" ] && echo "mathcode" || echo "allsplits")

SAVE_DIR="checkpoints/grpo_${MODEL_TAG}_cache_${DATA_TAG}_sft${SFT_COEF}_b${BETA}_c${CACHE_SIZE}_topk${CACHE_EXPERTS}_tmoeN${RATIO_N}_initsft_lr${LR}_dbg${NUM_STEPS}_rl${RL_COEF}_seq${PROMPT_LEN}-${COMPLETION_LEN}_n${MAX_SAMPLES}"
RUN_NAME="grpo-${MODEL_TAG}-cache-${DATA_TAG}-sft${SFT_COEF}-b${BETA}-c${CACHE_SIZE}-topk${CACHE_EXPERTS}-tmoeN${RATIO_N}-initsft-lr${LR}-dbg${NUM_STEPS}-rl${RL_COEF}-seq${PROMPT_LEN}-${COMPLETION_LEN}-n${MAX_SAMPLES}"

current_step() {  # $1 = SAVE_DIR -> prints global_step of its last checkpoint (0 if none)
    python3 -c "
from transformers.trainer_utils import get_last_checkpoint
import json
d = get_last_checkpoint('$1')
print(json.load(open(d + '/trainer_state.json'))['global_step']) if d else print(0)
" 2>/dev/null || echo 0
}

mkdir -p .wandb_run_ids
WANDB_ID_FILE=".wandb_run_ids/$(basename "$SAVE_DIR")"
if [ -f "$WANDB_ID_FILE" ]; then
    export WANDB_RUN_ID=$(cat "$WANDB_ID_FILE")
else
    export WANDB_RUN_ID=$(python3 -c "import wandb; print(wandb.util.generate_id())")
    echo "$WANDB_RUN_ID" > "$WANDB_ID_FILE"
fi
export WANDB_RESUME=allow

INIT_ARGS=""
if [ "$(current_step "$SAVE_DIR")" -eq 0 ]; then
    INIT_ARGS="--init-adapter $INIT_ADAPTER"
    echo "[chain] first cycle -- initializing from $INIT_ADAPTER"
fi

for ATTEMPT in 1 2 3; do
    START=$(date +%s)
    timeout 10200 accelerate launch \
        --multi_gpu \
        --num_processes 4 \
        scripts/finetune_moe_grpo.py \
        --model "$MODEL" \
        --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
        --dataset-split "$DATASET_SPLIT" \
        --max-samples "$MAX_SAMPLES" \
        --prompt-len "$PROMPT_LEN" \
        --completion-len "$COMPLETION_LEN" \
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
        --cache-topk \
        --temporal \
        --ratio-loss-N "$RATIO_N" \
        --lora-r 16 \
        --lora-alpha 32 \
        --seed 42 \
        --wandb-project moe-cache-reinforce \
        --wandb-run-name "$RUN_NAME" \
        --save-dir "$SAVE_DIR" \
        --save-every "$SAVE_EVERY" \
        --eval-ppl-every 10 \
        --resume \
        $INIT_ARGS && break
    ELAPSED=$(( $(date +%s) - START ))
    if [ $ELAPSED -gt 600 ]; then echo "[retry] failure after ${ELAPSED}s, not retrying"; break; fi
    echo "[retry] fast startup failure (attempt $ATTEMPT, ${ELAPSED}s), retrying in 60s..."
    sleep 60
done

STEP=$(current_step "$SAVE_DIR")
echo "[chain] progress: ${STEP}/${NUM_STEPS}"
if [ "$STEP" -lt "$NUM_STEPS" ]; then
    echo "[chain] not done -- resubmitting"
    sbatch scripts/run_finetune_moe_temporal_sft_then_dapo_chain.sh
else
    echo "[chain] complete"
fi
