#!/bin/bash
#SBATCH --job-name=smoke_test_context
#SBATCH --output=smoke_test_context_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00

# One-off memory/throughput smoke test: does batch-size 16, no gradient
# accumulation, fit on a SINGLE A100L at prompt+completion = 2048+2048
# tokens, for each of the four training scripts in this directory? Falls
# back to reporting 1024+1024 too, in case 2048+2048 doesn't fit.
#
# short-unkillable requires a minimum of 4 GPUs/job (QOS floor) even though
# only GPU 0 is actually used here -- same pattern as scripts/eval/
# run_benchmarks.sh. A handful of steps (--num-steps 3) on a small slice of
# the dataset is enough to see whether the forward/backward pass OOMs and
# to get a rough per-step wall-clock, without waiting out a real run.
#
# Usage: sbatch scripts/train/smoke_test_context.sh

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export WANDB_MODE=disabled  # diagnostic runs only, no real wandb run wanted
export CUDA_VISIBLE_DEVICES=0

SAVE_DIR="/tmp/smoke_test_context_${SLURM_JOB_ID}"
BATCH_SIZE=16

run_case() {
    # --logging-steps only exists on finetune_moe_grpo.py's argparse, not
    # controller/melinoe's -- pass it separately, only when the script wants it.
    local label="$1" script="$2" plen="$3" clen="$4"; shift 4
    echo "=== $label: prompt_len=$plen completion_len=$clen batch_size=$BATCH_SIZE ==="
    local start=$(date +%s)
    TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}_${label}" \
        python "$script" \
        --dataset nvidia/Nemotron-Post-Training-Dataset-v2 --dataset-split math,code \
        --max-samples 64 --prompt-len "$plen" --completion-len "$clen" \
        --batch-size "$BATCH_SIZE" --gradient-accumulation-steps 1 --num-steps 3 \
        --save-dir "${SAVE_DIR}_${label}" --save-every 1000000 \
        "$@" > "smoke_${label}.log" 2>&1 &
    local pid=$!

    # Poll GPU 0's memory.used while the process runs -- it's freed the
    # instant the process exits, so this has to be sampled live, not
    # queried after the fact (torch.cuda.max_memory_allocated in a fresh
    # process after exit would just read 0, a different process's counter).
    local peak_mb=0
    while kill -0 "$pid" 2>/dev/null; do
        local cur=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
        if [ -n "$cur" ] && [ "$cur" -gt "$peak_mb" ]; then peak_mb=$cur; fi
        sleep 1
    done
    wait "$pid"
    local status=$?
    local elapsed=$(( $(date +%s) - start ))
    if [ $status -eq 0 ]; then
        echo "  OK -- ${elapsed}s for 3 steps ($(( elapsed / 3 ))s/step), peak GPU0 mem ${peak_mb} MiB"
    elif grep -qi "CUDA out of memory\|OutOfMemoryError" "smoke_${label}.log"; then
        echo "  OOM after ${elapsed}s, peak GPU0 mem ${peak_mb} MiB -- see smoke_${label}.log"
    else
        echo "  FAILED (exit $status) after ${elapsed}s, peak GPU0 mem ${peak_mb} MiB -- see smoke_${label}.log"
    fi
}

# cache_sft is the most memory-hungry variant to test (on-policy generation
# up to completion_len tokens, plus the SFT NLL forward pass) -- if this
# fits, cache_reward (same architecture, --sft-coef 0, strictly less memory)
# fits too, so it isn't tested separately here.
GRPO_ARGS="--logging-steps 1 --num-generations 8 --rl-coef 2.0 --sft-coef 0.5 --beta 0.08 --cache-size 4 --cache-layer -1 --cache-experts-per-token 2 --cache-topk --soft-cache"
run_case cache_sft_2048 scripts/train/finetune_moe_grpo.py 2048 2048 $GRPO_ARGS
run_case cache_sft_1024 scripts/train/finetune_moe_grpo.py 1024 1024 $GRPO_ARGS

CONTROLLER_ARGS="--cache-size 4 --cache-layer -1 --deliberation-cost 0.02"
run_case controller_2048 scripts/train/finetune_moe_controller.py 2048 2048 $CONTROLLER_ARGS
run_case controller_1024 scripts/train/finetune_moe_controller.py 1024 1024 $CONTROLLER_ARGS

MELINOE_ARGS="--cache-size 4 --cache-layer -1"
run_case melinoe_2048 scripts/train/finetune_moe_melinoe.py 2048 2048 $MELINOE_ARGS
run_case melinoe_1024 scripts/train/finetune_moe_melinoe.py 1024 1024 $MELINOE_ARGS

rm -rf "${SAVE_DIR}"_*
echo "[smoke_test_context] done"
