#!/bin/bash
#SBATCH --job-name=qwen_finetune
#SBATCH --output=qwen_finetune_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --gres=gpu:80gb:4
#SBATCH --partition=long
#SBATCH --time=12:00:00

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FSDP_FLAGS="--use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap Qwen2MoeDecoderLayer \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_offload_params true \
    --fsdp_use_orig_params true"

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
