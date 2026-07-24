#!/bin/bash
#SBATCH --job-name=sft_baseline
#SBATCH --output=sft_baseline_%j.out
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

# Pin the wandb run id (shared-filesystem file keyed by SLURM job ID, not
# /tmp -- --requeue can land the same job ID on a different node) so a
# preemption + automatic requeue continues the same wandb run.
mkdir -p .wandb_run_ids
WANDB_ID_FILE=".wandb_run_ids/${SLURM_JOB_ID}"
if [ -f "$WANDB_ID_FILE" ]; then
    export WANDB_RUN_ID=$(cat "$WANDB_ID_FILE")
else
    export WANDB_RUN_ID=$(python3 -c "import wandb; print(wandb.util.generate_id())")
    echo "$WANDB_RUN_ID" > "$WANDB_ID_FILE"
fi
export WANDB_RESUME=allow

MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
LR="${LR:-1e-4}"
NUM_STEPS="${NUM_STEPS:-150}"
PROMPT_LEN="${PROMPT_LEN:-512}"
COMPLETION_LEN="${COMPLETION_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DATASET_SPLIT="${DATASET_SPLIT:-math,code}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"

DATA_TAG=$([ "$DATASET_SPLIT" = "math,code" ] && echo "mathcode" || echo "allsplits")
SEQ_TAG=""
if [ "$PROMPT_LEN" != "512" ] || [ "$COMPLETION_LEN" != "512" ]; then
    SEQ_TAG="_seq${PROMPT_LEN}-${COMPLETION_LEN}"
fi
if [ "$MAX_SAMPLES" != "20000" ]; then SEQ_TAG="${SEQ_TAG}_n${MAX_SAMPLES}"; fi
SAVE_DIR="checkpoints/sft_${MODEL_TAG}_${DATA_TAG}_lr${LR}${SEQ_TAG}"
RUN_NAME="sft-${MODEL_TAG}-${DATA_TAG}-lr${LR}${SEQ_TAG}"

# Retry fast startup failures (shared-FS flakiness: triton JIT getsource
# errors, NCCL rendezvous timeouts, HF cache lock contention). A failure
# after >10 min is real.
for ATTEMPT in 1 2 3; do
    START=$(date +%s)
    accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    scripts/finetune_moe_sft.py \
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
    --lora-r 16 \
    --lora-alpha 32 \
    --seed 42 \
    --wandb-project moe-cache-reinforce \
    --wandb-run-name "$RUN_NAME" \
    --save-dir "$SAVE_DIR" \
    --save-every 50 \
    --resume && break
    ELAPSED=$(( $(date +%s) - START ))
    if [ $ELAPSED -gt 600 ]; then echo "[retry] failure after ${ELAPSED}s, not retrying"; break; fi
    echo "[retry] fast startup failure (attempt $ATTEMPT, ${ELAPSED}s), retrying in 60s..."
    sleep 60
done
[ "$ATTEMPT" = "3" ] && [ ! -d "$SAVE_DIR" ] && { echo "[retry] all attempts failed, no checkpoint saved"; exit 1; }
exit 0
