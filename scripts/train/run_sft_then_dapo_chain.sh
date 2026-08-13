#!/bin/bash
#SBATCH --job-name=grpo_sftinit_chain
#SBATCH --output=grpo_sftinit_chain_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120

# Two-stage variant: (1) full SFT with LoRA [already done -- reuses the
# completed sft_baseline checkpoint as-is], (2) DAPO RL (soft-cache reward)
# initialized from that SFT adapter's weights via --init-adapter, instead of
# starting RL from a fresh LoRA on the raw base model. KL(policy||ref) still
# regularizes against the frozen BASE model (ref = adapters disabled), not
# the SFT checkpoint -- --init-adapter only seeds the starting point, it
# doesn't change what "ref" means. BETA is raised to 0.1 (vs 0.08 in the
# from-scratch cache_sft run) since there's now real SFT quality worth
# protecting from RL drift. Perplexity on the held-out SFT-style eval set is
# tracked the same way as before (--eval-ppl-every, PerplexityCallback).
#
# One run only (not two), so this uses the full 4-GPU allocation directly
# (no 2+2 split). Resubmits itself via `sbatch` if NUM_STEPS isn't reached
# within the 3h window; --signal=B:USR1@120 + the training script's own
# Preemption handler checkpoint before SLURM kills the job at the limit.

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
BETA="${BETA:-0.1}"
RL_COEF="${RL_COEF:-2.0}"
SFT_COEF="${SFT_COEF:-0.5}"
CACHE_SIZE="${CACHE_SIZE:-4}"
CACHE_LAYER="${CACHE_LAYER:--1}"
CACHE_EXPERTS="${CACHE_EXPERTS:-2}"
SAVE_EVERY="${SAVE_EVERY:-10}"
INIT_ADAPTER="${INIT_ADAPTER:-checkpoints/sft_phi-tiny-moe-instruct_allsplits_lr1e-4_seq1024-1024_n2000/checkpoint-32}"
DATA_TAG=$([ "$DATASET_SPLIT" = "math,code" ] && echo "mathcode" || echo "allsplits")

SAVE_DIR="checkpoints/grpo_${MODEL_TAG}_cache_${DATA_TAG}_sft${SFT_COEF}_b${BETA}_c${CACHE_SIZE}_softall_initsft_lr${LR}_dbg${NUM_STEPS}_rl${RL_COEF}_seq${PROMPT_LEN}-${COMPLETION_LEN}_n${MAX_SAMPLES}"
RUN_NAME="grpo-${MODEL_TAG}-cache-${DATA_TAG}-sft${SFT_COEF}-b${BETA}-c${CACHE_SIZE}-softall-initsft-lr${LR}-dbg${NUM_STEPS}-rl${RL_COEF}-seq${PROMPT_LEN}-${COMPLETION_LEN}-n${MAX_SAMPLES}"

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

# Wrapped in `timeout` (< the 3h SLURM cap) so this bash wrapper always
# regains control with buffer to spare and can run the resubmit logic below
# -- relying solely on --signal=B:USR1@120 + the training script's own
# graceful shutdown was NOT enough: if that shutdown (checkpoint save,
# process teardown) takes longer than the ~120s warning window, SLURM's
# hard time-limit kill takes out this whole wrapper script too, silently
# stalling the chain with no resubmission (observed: job 10193167 FAILED
# at the 3h mark, stuck at step 70/250, never resubmitted).
#
# Also retries fast startup failures (shared-FS flakiness: triton JIT
# getsource errors, NCCL rendezvous timeouts, HF cache lock contention --
# same pattern already handled in run_finetune_moe_grpo.sh). Observed once:
# job 10207804 crashed ~77s in on a transient "could not get source code"
# triton error, ate a full resubmit cycle for zero progress since there was
# no retry here yet.
for ATTEMPT in 1 2 3; do
    START=$(date +%s)
    timeout 10200 accelerate launch \
        --multi_gpu \
        --num_processes 4 \
        scripts/train/finetune_moe_grpo.py \
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
        --soft-cache \
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
    sbatch scripts/train/run_sft_then_dapo_chain.sh
else
    echo "[chain] complete"
fi
