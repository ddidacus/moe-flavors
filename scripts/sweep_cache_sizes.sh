#!/bin/bash
# Cache-size sweep: C in {2,4,8,12} of 16 experts (top-2 routing), all other
# knobs at the current best config (alpha=0.3, beta=0.08, normalize_then_sum).
# Each point is an independent sbatch of run_finetune_moe_grpo.sh; save dirs
# encode the config, so re-running this script resumes finished-wall jobs.
set -e
cd "$(dirname "$0")/.."

ALPHA="${ALPHA:-0.3}"
BETA="${BETA:-0.08}"
SIZES=(${SIZES:-2 4 8 12})

for C in "${SIZES[@]}"; do
    NAME="grpo_phi-tiny-moe-instruct_cache_mathcode_a${ALPHA}_b${BETA}_kds1.0_c${C}_topk2"
    if [ -f "checkpoints/$NAME/COMPLETED" ]; then
        echo "c=$C already completed, skipping"
        continue
    fi
    JOB=$(ALPHA=$ALPHA BETA=$BETA KD_SCALE=1.0 CACHE_SIZE=$C \
          sbatch --parsable scripts/run_finetune_moe_grpo.sh)
    echo "cache_size=$C -> job $JOB"
done
