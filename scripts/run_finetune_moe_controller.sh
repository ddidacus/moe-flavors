#!/bin/bash
#SBATCH --job-name=controller_baseline
#SBATCH --output=controller_baseline_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120

# Multi-GPU via accelerate/DDP (4x A100-80GB, matching the other three
# training scripts) -- NOT yet empirically validated at 4 GPUs (the one
# completed controller_baseline run so far was single-GPU, on a 48GB L40S).
# The script already uses HF Accelerate (accelerator.prepare(peft_model,
# controller, optimizer, dataloader)), which should DDP-wrap generically,
# but confirm the first run here doesn't hit a resume/checkpoint-sync issue
# across ranks before trusting it unattended.

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
NUM_STEPS="${NUM_STEPS:-150}"
CACHE_SIZE="${CACHE_SIZE:-4}"
DELIBERATION_COST="${DELIBERATION_COST:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-4}"
PROMPT_LEN="${PROMPT_LEN:-512}"
COMPLETION_LEN="${COMPLETION_LEN:-512}"
DATASET_SPLIT="${DATASET_SPLIT:-math,code}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"

DATA_TAG=$([ "$DATASET_SPLIT" = "math,code" ] && echo "mathcode" || echo "allsplits")
SEQ_TAG=""
if [ "$PROMPT_LEN" != "512" ] || [ "$COMPLETION_LEN" != "512" ]; then
    SEQ_TAG="_seq${PROMPT_LEN}-${COMPLETION_LEN}"
fi
if [ "$MAX_SAMPLES" != "20000" ]; then SEQ_TAG="${SEQ_TAG}_n${MAX_SAMPLES}"; fi
SAVE_DIR="checkpoints/controller_${MODEL_TAG}_${DATA_TAG}_c${CACHE_SIZE}_eta${DELIBERATION_COST}${SEQ_TAG}"
RUN_NAME="controller-${MODEL_TAG}-${DATA_TAG}-c${CACHE_SIZE}-eta${DELIBERATION_COST}${SEQ_TAG}"

accelerate launch \
--multi_gpu \
--num_processes 4 \
scripts/finetune_moe_controller.py \
--model "$MODEL" \
--dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
--dataset-split "$DATASET_SPLIT" \
--max-samples "$MAX_SAMPLES" \
--prompt-len "$PROMPT_LEN" \
--completion-len "$COMPLETION_LEN" \
--batch-size "$BATCH_SIZE" \
--gradient-accumulation-steps "$GRAD_ACCUM" \
--num-steps "$NUM_STEPS" \
--lr "$LR" \
--cache-size "$CACHE_SIZE" \
--cache-layer -1 \
--deliberation-cost "$DELIBERATION_COST" \
--lora-r 16 \
--lora-alpha 32 \
--seed 42 \
--wandb-project moe-cache-reinforce \
--wandb-run-name "$RUN_NAME" \
--save-dir "$SAVE_DIR" \
--save-every 50 \
--resume
