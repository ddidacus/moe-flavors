#!/bin/bash
# Submits eval_complete.py (all non-prompt_conditioned variants) and
# eval_complete_cache_conditioned.py (prompt_conditioned, cache-size swept)
# for both models' checkpoints, paired 2-per-job via run_pair.sh (each on
# its own GPU within tamia's always-4-H100-exclusive allocation, matching
# the "2 scripts/job, 1 GPU + 4 CPUs each" packing scheme -- 2 of tamia's
# 4 granted GPUs go unused per job, same tradeoff already accepted
# elsewhere for single/dual-GPU jobs on this cluster's node-exclusive QOS).
#
# All checkpoints for both models live only on tamia (this is where they
# were trained) -- mila has no local copies of the tamia-trained
# checkpoints, so the whole sweep runs here rather than splitting across
# clusters. Cross-job parallelism comes from tamia's scheduler running
# several of these jobs concurrently, not from a mila/tamia split.
#
# prompt_conditioned is NOT included in eval_complete.py's list (it
# expects a "[CACHE_SIZE=X]" prefix on every prompt at inference time --
# evaluating it unconditioned would be out-of-distribution for that
# checkpoint) -- it only appears via eval_complete_cache_conditioned.py.
#
# Usage: bash scripts/cluv/eval_complete_sweep_tamia.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PHI="microsoft/Phi-tiny-MoE-instruct"
OLMOE="allenai/OLMoE-1B-7B-0125-Instruct"
TIME="12:00:00"

# model, variant, checkpoint_dir, cache_size, experts_per_token
phi_combos=(
    "$PHI|base||4|2"
    "$PHI|sft_baseline|checkpoints/sft_baseline_tamia|4|2"
    "$PHI|cache_sft|checkpoints/cache_sft_tamia_v2|4|2"
    "$PHI|cache_reward|checkpoints/cache_reward_tamia|4|2"
    "$PHI|controller_baseline|checkpoints/controller_baseline_tamia_v2|4|2"
    "$PHI|melinoe_baseline|checkpoints/melinoe_baseline_tamia_v2|4|2"
)
olmoe_combos=(
    "$OLMOE|base||16|8"
    "$OLMOE|sft_baseline|checkpoints/sft_baseline_olmoe_tamia|16|8"
    "$OLMOE|cache_reward|checkpoints/cache_reward_olmoe_tamia|16|8"
    "$OLMOE|controller_baseline|checkpoints/controller_baseline_olmoe_tamia|16|8"
    "$OLMOE|melinoe_baseline|checkpoints/melinoe_baseline_olmoe_tamia|16|8"
)
all_combos=("${phi_combos[@]}" "${olmoe_combos[@]}")

build_cmd() {
    local model=$1 variant=$2 ckpt=$3 cache_size=$4 experts=$5
    local ckpt_args=""
    if [ -n "$ckpt" ]; then ckpt_args="--checkpoint-dir $ckpt"; fi
    # cluv already runs everything through `uv run --directory=moe-flavors`
    # (handles venv/cwd) and common.sh already sets HF_HOME/HF_HUB_OFFLINE --
    # no need to redo any of that here, only HF_ALLOW_CODE_EVAL (HumanEval's
    # code_eval self-test import-time check, not set by common.sh).
    echo "HF_ALLOW_CODE_EVAL=1 python scripts/eval/eval_complete.py --model $model --variant $variant $ckpt_args --cache-size $cache_size --cache-experts-per-token $experts --cache-topk"
}

n=${#all_combos[@]}
i=0
while [ $i -lt $n ]; do
    IFS='|' read -r m1 v1 c1 cs1 e1 <<< "${all_combos[$i]}"
    cmd1=$(build_cmd "$m1" "$v1" "$c1" "$cs1" "$e1")
    cmd2="" v2=""
    if [ $((i + 1)) -lt $n ]; then
        IFS='|' read -r m2 v2 c2 cs2 e2 <<< "${all_combos[$((i + 1))]}"
        cmd2=$(build_cmd "$m2" "$v2" "$c2" "$cs2" "$e2")
    fi
    echo "=== job: $v1 (+ ${v2:-none}) ==="
    cluv submit --autocommit tamia scripts/cluv/tamia_job.sh --time="$TIME" -- \
        bash scripts/eval/run_pair.sh "$cmd1" "$cmd2"
    i=$((i + 2))
done

# --- cache-size-conditioned copy, one job per model (own job, not paired --
# each already sweeps 3 cache sizes internally, long-running on its own) ---
cc_cmd() {
    local model=$1 ckpt=$2 sizes=$3 experts=$4
    echo "HF_ALLOW_CODE_EVAL=1 python scripts/eval/eval_complete_cache_conditioned.py --model $model --checkpoint-dir $ckpt --cache-sizes $sizes --cache-experts-per-token $experts --cache-topk"
}

echo "=== job: phi prompt_conditioned (cache-conditioned sweep) ==="
cluv submit --autocommit tamia scripts/cluv/tamia_job.sh --time="$TIME" -- \
    bash scripts/eval/run_pair.sh "$(cc_cmd "$PHI" checkpoints/prompt_conditioned_tamia_200steps 2,4,8 2)"

echo "=== job: olmoe prompt_conditioned (cache-conditioned sweep) ==="
cluv submit --autocommit tamia scripts/cluv/tamia_job.sh --time="$TIME" -- \
    bash scripts/eval/run_pair.sh "$(cc_cmd "$OLMOE" checkpoints/prompt_conditioned_olmoe_tamia 8,16,32 8)"

echo "sweep submitted."
