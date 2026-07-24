#!/bin/bash
# Submit a soft-cache (LRU cache-hit-rate) eval via cluv: one job, up to 5
# variants in parallel (GPU-pinned), targeting checkpoints/<variant>_<CLUSTER>
# -- the naming scripts/cluv/train_<variant>.sh saves to. See
# scripts/eval_soft_cache.py's module docstring for what this does and does
# NOT implement yet (it's a stub -- raw cache-hit-rate numbers only, no
# routing-distribution plots).
#
# Usage: CLUSTER=fir bash scripts/cluv/eval_soft_cache.sh [variant ...]
#        (default: base sft_baseline cache_sft temporal_moe controller_baseline)
#
# Pass at most 4 variants at once (the job requests one exclusive 4-GPU
# node); results land in evals/<CLUSTER>/soft_cache_<date>/results_soft_cache_<variant>.json
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
    VARIANTS=(base sft_baseline cache_sft temporal_moe controller_baseline)
fi

cluv submit --autocommit "$CLUSTER" -- bash scripts/cluv/_eval_soft_cache_run.sh "$CLUSTER" "${VARIANTS[@]}"
echo "eval_soft_cache [${VARIANTS[*]}] -> submitted to $CLUSTER"
