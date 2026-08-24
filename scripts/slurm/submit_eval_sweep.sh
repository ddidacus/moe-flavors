#!/bin/bash
# Plain-SLURM equivalent of scripts/cluv/eval_complete_sweep_tamia.sh --
# submits the full eval_complete.py sweep (every trained variant, both
# models) via `sbatch` directly, no cluv. Edit the checkpoint paths below
# to match wherever your checkpoints actually live, and edit
# scripts/slurm/eval_complete_pair.sbatch's #SBATCH lines for your
# cluster's partition/account.
#
# prompt_conditioned is handled separately via eval_complete_cache_
# conditioned.py (it expects a "[CACHE_SIZE=X]" prefix on every prompt,
# so evaluating it unconditioned like the rest would be out of
# distribution for that checkpoint) -- see submit_eval_sweep_cache_
# conditioned.sh.
#
# Usage: bash scripts/slurm/submit_eval_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PHI="microsoft/Phi-tiny-MoE-instruct"
OLMOE="allenai/OLMoE-1B-7B-0125-Instruct"
JOBS_DIR="scripts/eval/_sweep_jobs"
mkdir -p "$JOBS_DIR" logs/slurm

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
    echo "=== job $job_num: $v1 (+ $v2) ==="
    sbatch scripts/slurm/eval_complete_pair.sbatch "$f1" "$f2"
    i=$((i + 2))
done

echo "sweep submitted."
