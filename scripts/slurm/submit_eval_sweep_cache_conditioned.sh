#!/bin/bash
# Plain-SLURM equivalent of the cache-conditioned half of scripts/cluv/
# eval_complete_sweep_tamia.sh -- submits eval_complete_cache_conditioned.py
# for the prompt_conditioned checkpoint of each model, swept across every
# cache size it was trained on. No cluv, just `sbatch` directly.
#
# Usage: bash scripts/slurm/submit_eval_sweep_cache_conditioned.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PHI="microsoft/Phi-tiny-MoE-instruct"
OLMOE="allenai/OLMoE-1B-7B-0125-Instruct"
JOBS_DIR="scripts/eval/_sweep_jobs"
mkdir -p "$JOBS_DIR" logs/slurm

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

f_phi_cc="$JOBS_DIR/job_cc_phi_prompt_conditioned.sh"
write_cc_job "$f_phi_cc" "$PHI" checkpoints/prompt_conditioned_tamia_200steps 2,4,8 2
echo "=== job: phi prompt_conditioned (cache-conditioned sweep) ==="
sbatch scripts/slurm/eval_complete_pair.sbatch "$f_phi_cc"

f_olmoe_cc="$JOBS_DIR/job_cc_olmoe_prompt_conditioned.sh"
write_cc_job "$f_olmoe_cc" "$OLMOE" checkpoints/prompt_conditioned_olmoe_tamia 8,16,32 8
echo "=== job: olmoe prompt_conditioned (cache-conditioned sweep) ==="
sbatch scripts/slurm/eval_complete_pair.sbatch "$f_olmoe_cc"

echo "sweep submitted."
