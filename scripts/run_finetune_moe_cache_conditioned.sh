#!/bin/bash
#SBATCH --job-name=cache_conditioned_chain
#SBATCH --output=cache_conditioned_chain_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120

# cache_conditioned variant: same base model + loss as cache_sft (DAPO policy
# loss + SFT NLL, see finetune_moe_grpo.py's GRPOTrainerWithSFT), but the
# router at --cache-layer is conditioned on a TARGET cache size via a
# sinusoidal embedding added to its input hidden states (--conditioned-
# cache-sizes), round-robin cycled every step so each candidate size gets
# equal interleaved exposure -- 180 steps / 3 sizes = 60 steps = 3840
# samples each at effective batch 64 (8 * 2 grad-accum * 4 GPUs). The LRU
# capacity simulated for the reward always matches the CURRENT step's
# conditioned size (see RewardEngine.cache_size_state in finetune_moe_grpo.py).
#
# Same chaining mechanics as run_finetune_moe_sft_then_dapo_chain.sh:
# timeout 10200 (< 3h SLURM cap) so the wrapper always regains control to
# run its own resubmit logic, plus a 3-attempt retry loop for transient
# shared-FS/triton-JIT startup failures. cache_sft-scale runs (~142.6s/step
# per train_small_scale.sh's own estimate) take ~7h for 180 steps, so
# multiple resubmit cycles are expected.

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
NUM_STEPS="${NUM_STEPS:-180}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
BETA="${BETA:-0.08}"
RL_COEF="${RL_COEF:-2.0}"
SFT_COEF="${SFT_COEF:-0.5}"
CACHE_LAYER="${CACHE_LAYER:--1}"
CACHE_EXPERTS="${CACHE_EXPERTS:-2}"
CONDITIONED_CACHE_SIZES="${CONDITIONED_CACHE_SIZES:-2,4,8}"
SAVE_EVERY="${SAVE_EVERY:-10}"
INIT_ADAPTER="${INIT_ADAPTER:-}"
DATA_TAG=$([ "$DATASET_SPLIT" = "math,code" ] && echo "mathcode" || echo "allsplits")
SIZES_TAG=$(echo "$CONDITIONED_CACHE_SIZES" | tr ',' '-')

SAVE_DIR="checkpoints/cachecond_${MODEL_TAG}_${DATA_TAG}_sft${SFT_COEF}_b${BETA}_sizes${SIZES_TAG}_lr${LR}_dbg${NUM_STEPS}_rl${RL_COEF}_seq${PROMPT_LEN}-${COMPLETION_LEN}_n${MAX_SAMPLES}"
RUN_NAME="cachecond-${MODEL_TAG}-${DATA_TAG}-sft${SFT_COEF}-b${BETA}-sizes${SIZES_TAG}-lr${LR}-dbg${NUM_STEPS}-rl${RL_COEF}-seq${PROMPT_LEN}-${COMPLETION_LEN}-n${MAX_SAMPLES}"

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
export WANDB_TAGS="${WANDB_TAGS:-small_scale}"

INIT_ARGS=""
if [ -n "$INIT_ADAPTER" ] && [ "$(current_step "$SAVE_DIR")" -eq 0 ]; then
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
        --cache-layer "$CACHE_LAYER" \
        --cache-experts-per-token "$CACHE_EXPERTS" \
        --cache-topk \
        --conditioned-cache-sizes "$CONDITIONED_CACHE_SIZES" \
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
    sbatch scripts/run_finetune_moe_cache_conditioned.sh
else
    echo "[chain] complete"
fi
