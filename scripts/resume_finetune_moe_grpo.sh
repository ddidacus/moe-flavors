#!/bin/bash
#SBATCH --job-name=grpo_resume
#SBATCH --output=grpo_resume_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=long
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

# Long-running resume of an existing GRPO save-dir to a higher --num-steps,
# on the preemptible `long` partition. Unlike run_finetune_moe_grpo.sh (which
# derives SAVE_DIR from tags and always starts a fresh run), this script
# always resumes a SPECIFIC checkpoint dir you pass in, and is safe against
# preemption:
#   - --requeue + --signal=B:USR1@120: SLURM warns the job 120s before
#     killing it (time limit or preemption by a higher-priority job), then
#     automatically resubmits the SAME job id to run again later.
#   - PreemptionCallback (scripts/finetune_moe_grpo.py) catches that signal
#     (or a plain SIGTERM) and checkpoints + stops cleanly at the next step
#     boundary -- optimizer/scheduler/RNG state included, not just weights.
#   - On restart, --resume + the same --save-dir finds that checkpoint via
#     get_last_checkpoint() and continues exactly where it left off.
#   - WANDB_RUN_ID + WANDB_RESUME=allow make every restart append to the
#     SAME wandb run instead of starting a new one.
#
# Required env vars: SAVE_DIR (existing checkpoint dir to resume),
#   WANDB_RUN_ID (the run to keep appending to), RUN_NAME (for display).
# Everything else mirrors the exact hyperparameters that checkpoint was
# trained under -- override via env vars only if you're sure the new value
# is compatible with the existing optimizer/scheduler state.

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

SAVE_DIR="${SAVE_DIR:?set SAVE_DIR to the existing checkpoint dir to resume}"
WANDB_RUN_ID="${WANDB_RUN_ID:?set WANDB_RUN_ID to the wandb run to keep appending to}"
RUN_NAME="${RUN_NAME:?set RUN_NAME (display name, cosmetic -- WANDB_RUN_ID governs continuity)}"
NUM_STEPS="${NUM_STEPS:-2000}"
export WANDB_RUN_ID
export WANDB_RESUME=allow

MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
LR="${LR:-1e-4}"
BETA="${BETA:-0.08}"
RL_COEF="${RL_COEF:-2.0}"
SFT_COEF="${SFT_COEF:-0.5}"
CACHE_SIZE="${CACHE_SIZE:-4}"
CACHE_LAYER="${CACHE_LAYER:--1}"
NUM_GEN="${NUM_GEN:-8}"

accelerate launch \
--multi_gpu \
--num_processes 4 \
scripts/finetune_moe_grpo.py \
--model "$MODEL" \
--dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
--dataset-split "math,code" \
--max-samples 20000 \
--prompt-len 512 \
--completion-len 512 \
--num-generations "$NUM_GEN" \
--batch-size 16 \
--gradient-accumulation-steps 1 \
--num-steps "$NUM_STEPS" \
--num-epochs 10 \
--lr "$LR" \
--temperature 1.0 \
--rl-coef "$RL_COEF" \
--sft-coef "$SFT_COEF" \
--beta "$BETA" \
--cache-size "$CACHE_SIZE" \
--cache-layer "$CACHE_LAYER" \
--cache-experts-per-token 2 \
--cache-topk \
--soft-cache \
--lora-r 16 \
--lora-alpha 32 \
--seed 42 \
--wandb-project moe-cache-reinforce \
--wandb-run-name "$RUN_NAME" \
--save-dir "$SAVE_DIR" \
--save-every 50 \
--eval-ppl-every 10 \
--resume
