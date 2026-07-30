#!/bin/bash
# Submit an lm-eval-harness eval via cluv: one job, up to 5 variants in
# parallel (GPU-pinned), mirroring scripts/run_eval_lm_harness.sh's
# pattern but targeting checkpoints/<variant>_<CLUSTER> -- the naming
# scripts/cluv/train_<variant>.sh saves to.
#
# Usage: CLUSTER=fir bash scripts/cluv/eval_lm_harness.sh [variant ...]
#        (default: base sft_baseline cache_sft temporal_moe controller_baseline)
#
# Pass at most 4 variants at once (the job requests one exclusive 4-GPU
# node); results land in evals/<CLUSTER>/<date>/results_<variant>.json --
# merge them afterward with:
#   cluv submit <cluster> -- python scripts/eval_lm_harness.py --variant merge --out-dir evals/<cluster>/<date>
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER="${CLUSTER:?set CLUSTER=<tamia|rorqual|narval|vulcan|fir|nibi|first>}"

VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
    VARIANTS=(base sft_baseline cache_sft temporal_moe controller_baseline)
fi

cluv submit --autocommit "$CLUSTER" -- bash scripts/cluv/_eval_lm_harness_run.sh "$CLUSTER" "${VARIANTS[@]}"
echo "eval_lm_harness [${VARIANTS[*]}] -> submitted to $CLUSTER"
