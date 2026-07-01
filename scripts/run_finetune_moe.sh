#!/bin/bash
#SBATCH --job-name=ft_gptoss
#SBATCH --output=ft_gptoss_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --gres=gpu:80gb:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --signal=B:USR1@120

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

# --- Configuration ---
# Set TEMPORAL=1 to enable temporal boundary routing on existing MoE experts
TEMPORAL="${TEMPORAL:-0}"

if [ "$TEMPORAL" = "1" ]; then
    SAVE_DIR="checkpoints/finetune_gptoss_lora_temporal"
    RUN_NAME="finetune-gptoss-20b-lora-temporal"
    EXTRA_ARGS="--temporal --ratio-loss-N 8 --ratio-loss-alpha 0.03 --entropy-threshold 0.1 --entropy-alpha 0.05 --entropy-warmup-steps 500"
else
    SAVE_DIR="checkpoints/finetune_gptoss_lora"
    RUN_NAME="finetune-gptoss-20b-lora"
    EXTRA_ARGS=""
fi

FSDP_FLAGS="--use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap GptOssDecoderLayer \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_offload_params true \
    --fsdp_use_orig_params true"

accelerate launch \
    $FSDP_FLAGS \
    --num_processes 4 \
    --mixed_precision bf16 \
    scripts/finetune_moe.py \
    --model openai/gpt-oss-20b \
    --dataset ddidacus/nemotron-moe-exam \
    --seq-len 4096 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --num-epochs 1 \
    --num-steps 500 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.03 \
    --lr-scheduler cosine \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --log-every 10 \
    --eval-every 250 \
    --seed 42 \
    --wandb-project moe-chunking-poc \
    --wandb-run-name "$RUN_NAME" \
    --save-dir "$SAVE_DIR" \
    --save-every 50 \
    --resume-from auto \
    $EXTRA_ARGS
