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
# Each (model,variant) command is written to its own tiny script file under
# scripts/eval/_sweep_jobs/ rather than passed as an inline string --
# cluv's own `submit ... -- program args` forwarding mangles long
# multi-flag command strings (reproduced: a short "echo hi" survived, the
# real ~250-char eval_complete.py invocation arrived as a silent no-op).
# File paths are simple tokens with no quoting risk. These job files are
# committed to git BEFORE any submission -- cluv's --autocommit only
# commits already-tracked files, so a fresh file needs an explicit add
# first or the remote job fails with "No such file or directory" (same
# trap hit earlier this session with train_prompt_conditioned.py).
#
# Usage: bash scripts/cluv/eval_complete_sweep_tamia.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PHI="microsoft/Phi-tiny-MoE-instruct"
OLMOE="allenai/OLMoE-1B-7B-0125-Instruct"
TIME="24:00:00"
JOBS_DIR="scripts/eval/_sweep_jobs"
mkdir -p "$JOBS_DIR"
rm -f "$JOBS_DIR"/*.sh

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

write_job() {
    local out=$1 model=$2 variant=$3 ckpt=$4 cache_size=$5 experts=$6
    local ckpt_args=""
    if [ -n "$ckpt" ]; then ckpt_args="--checkpoint-dir $ckpt"; fi
    cat > "$out" <<SCRIPT
#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
python scripts/eval/eval_complete.py --model $model --variant $variant $ckpt_args \\
    --cache-size $cache_size --cache-experts-per-token $experts --cache-topk
SCRIPT
}

write_cc_job() {
    local out=$1 model=$2 ckpt=$3 sizes=$4 experts=$5
    cat > "$out" <<SCRIPT
#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
python scripts/eval/eval_complete_cache_conditioned.py --model $model \\
    --checkpoint-dir $ckpt --cache-sizes $sizes \\
    --cache-experts-per-token $experts --cache-topk
SCRIPT
}

# --- pass 1: write every job-pair's two script files, recording the pairs ---
pairs=()  # each entry: "f1|f2|label"
n=${#all_combos[@]}
i=0
job_num=0
while [ $i -lt $n ]; do
    IFS='|' read -r m1 v1 c1 cs1 e1 <<< "${all_combos[$i]}"
    job_num=$((job_num + 1))
    f1="$JOBS_DIR/job${job_num}_a_${v1}.sh"
    write_job "$f1" "$m1" "$v1" "$c1" "$cs1" "$e1"
    f2="" v2="none"
    if [ $((i + 1)) -lt $n ]; then
        IFS='|' read -r m2 v2 c2 cs2 e2 <<< "${all_combos[$((i + 1))]}"
        f2="$JOBS_DIR/job${job_num}_b_${v2}.sh"
        write_job "$f2" "$m2" "$v2" "$c2" "$cs2" "$e2"
    fi
    pairs+=("$f1|$f2|$v1 (+ $v2)")
    i=$((i + 2))
done

f_phi_cc="$JOBS_DIR/job_cc_phi_prompt_conditioned.sh"
write_cc_job "$f_phi_cc" "$PHI" checkpoints/prompt_conditioned_tamia_200steps 2,4,8 2
pairs+=("$f_phi_cc||phi prompt_conditioned (cache-conditioned sweep)")

f_olmoe_cc="$JOBS_DIR/job_cc_olmoe_prompt_conditioned.sh"
write_cc_job "$f_olmoe_cc" "$OLMOE" checkpoints/prompt_conditioned_olmoe_tamia 8,16,32 8
pairs+=("$f_olmoe_cc||olmoe prompt_conditioned (cache-conditioned sweep)")

# --- commit every generated job file BEFORE any submission ---
git add "$JOBS_DIR"
git commit -m "Regenerate eval_complete_sweep_tamia.sh job files" --allow-empty-message -q || true

# --- pass 2: submit each pair ---
for entry in "${pairs[@]}"; do
    IFS='|' read -r f1 f2 label <<< "$entry"
    echo "=== job: $label ==="
    cluv submit --autocommit tamia scripts/cluv/tamia_job.sh --time="$TIME" -- \
        bash scripts/eval/run_pair.sh "$f1" "$f2"
done

echo "sweep submitted."
