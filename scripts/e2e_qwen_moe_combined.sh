#!/bin/bash
#SBATCH --job-name=qwen_moe_combined
#SBATCH --output=qwen_moe_combined_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --gres=gpu:80gb:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --exclude=cn-g011

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

FSDP_FLAGS="--use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap Qwen2MoeDecoderLayer \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_offload_params true \
    --fsdp_use_orig_params true"

# ── 1. Plain fine-tune (all 4 GPUs) ──

if [ ! -f checkpoints/finetune_qwen1.5_moe_a2.7b/COMPLETED ]; then
    echo "=== Starting fine-tune ==="

    accelerate launch \
        $FSDP_FLAGS \
        --num_processes 4 \
        --mixed_precision bf16 \
        scripts/finetune_qwen_moe.py \
        --model Qwen/Qwen1.5-MoE-A2.7B \
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
        --log-every 10 \
        --eval-every 250 \
        --seed 42 \
        --wandb-project moe-chunking-poc \
        --wandb-run-name finetune-qwen1.5-moe-a2.7b \
        --save-dir checkpoints/finetune_qwen1.5_moe_a2.7b \
        --save-every 10 \
        --resume-from auto

    echo "Fine-tune exited with status $?"
else
    echo "=== Fine-tune already completed, skipping ==="
fi

# ── 2. Temporal-wrap (all 4 GPUs) ──

if [ ! -f checkpoints/temporal_wrap_qwen1.5_moe/COMPLETED ]; then
    echo "=== Starting temporal-wrap ==="

    accelerate launch \
        $FSDP_FLAGS \
        --num_processes 4 \
        --mixed_precision bf16 \
        scripts/moe_mixin_poc.py \
        --moe-type temporal-wrap \
        --model Qwen/Qwen1.5-MoE-A2.7B \
        --ratio-loss-N 128 \
        --ratio-loss-alpha 0.03 \
        --entropy-threshold 0.1 \
        --entropy-alpha 0.05 \
        --entropy-warmup-steps 500 \
        --seq-len 4096 \
        --batch-size 1 \
        --gradient-accumulation-steps 16 \
        --num-epochs 1 \
        --num-steps 500 \
        --lr 2e-5 \
        --weight-decay 0.01 \
        --warmup-ratio 0.03 \
        --lr-scheduler cosine \
        --log-every 10 \
        --eval-every 250 \
        --seed 42 \
        --wandb-project moe-chunking-poc \
        --wandb-run-name temporal-wrap-qwen1.5-moe-a2.7b \
        --save-dir checkpoints/temporal_wrap_qwen1.5_moe \
        --save-every 10 \
        --resume-from auto

    echo "Temporal-wrap exited with status $?"
else
    echo "=== Temporal-wrap already completed, skipping ==="
fi
