#!/bin/bash
# Alpha sweep at fixed cache size: reward = (1-a)*cache + a*KD under
# normalize_then_sum (weights = exact influence ratios), beta=0.08.
# Save dirs encode the config; re-running resumes wall-hit jobs.
set -e
cd "$(dirname "$0")/.."

BETA="${BETA:-0.08}"
CACHE_SIZE="${CACHE_SIZE:-4}"
ALPHAS=(${ALPHAS:-0.1 0.3 0.5 0.7})

for A in "${ALPHAS[@]}"; do
    NAME="grpo_phi-tiny-moe-instruct_cache_mathcode_a${A}_b${BETA}_kds1.0_c${CACHE_SIZE}_topk2"
    if [ -f "checkpoints/$NAME/COMPLETED" ]; then
        echo "alpha=$A already completed, skipping"
        continue
    fi
    JOB=$(ALPHA=$A BETA=$BETA KD_SCALE=1.0 CACHE_SIZE=$CACHE_SIZE \
          sbatch --parsable scripts/run_finetune_moe_grpo.sh)
    echo "alpha=$A -> job $JOB"
done
