#!/bin/bash
#SBATCH --job-name=grpo_olmoe_cache
#SBATCH --output=grpo_olmoe_cache_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

# --- Configuration ---
MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
ALPHA="${ALPHA:-0}"          # weight of the r^BC reward func; prefer BETA
BETA="${BETA:-0.04}"         # TRL KL(policy||frozen base) coefficient
KD_SCALE="${KD_SCALE:-1.0}"  # multiplier on the KD reward (rl_moe reward_scale)
TEMPORAL="${TEMPORAL:-0}"    # 1: boundary-prediction + hold/switch routing mixin
RATIO_N="${RATIO_N:-8}"      # temporal target segment length
CACHE_SIZE="${CACHE_SIZE:-4}"    # of 16 experts (Phi-tiny); OLMoE: use 16 of 64
CACHE_LAYER="${CACHE_LAYER:--1}"   # -1 = middle layer
NUM_GEN="${NUM_GEN:-8}"      # G: group size (completions per prompt)
CACHE_TOPK="${CACHE_TOPK:-1}"    # 1: deterministic top-k expert reward (deployment-faithful)
CACHE_EXPERTS="${CACHE_EXPERTS:-2}"  # experts scored per token (= router top-k)

REWARD_TAG=$([ "$CACHE_TOPK" = "1" ] && echo "topk${CACHE_EXPERTS}" || echo "sampled${CACHE_EXPERTS}")
if [ "$TEMPORAL" = "1" ]; then REWARD_TAG="${REWARD_TAG}_tmoeN${RATIO_N}"; fi
SAVE_DIR="checkpoints/grpo_${MODEL_TAG}_cache_mathcode_a${ALPHA}_b${BETA}_kds${KD_SCALE}_c${CACHE_SIZE}_${REWARD_TAG}"
RUN_NAME="grpo-${MODEL_TAG}-cache-mathcode-a${ALPHA}-b${BETA}-kds${KD_SCALE}-c${CACHE_SIZE}-${REWARD_TAG}"
EXTRA_ARGS=$([ "$CACHE_TOPK" = "1" ] && echo "--cache-topk")
if [ "$TEMPORAL" = "1" ]; then EXTRA_ARGS="$EXTRA_ARGS --temporal --ratio-loss-N $RATIO_N"; fi

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
    --num-steps 500 \
    --num-epochs 10 \
    --lr 3e-5 \
    --temperature 1.0 \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --kd-scale "$KD_SCALE" \
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
    --resume \
    $EXTRA_ARGS
