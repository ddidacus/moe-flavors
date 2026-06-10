#!/bin/bash
#SBATCH --job-name=routing_analysis
#SBATCH --output=routing_analysis_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:4
#SBATCH --partition=long
#SBATCH --time=3:00:00

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1

SAVE_DIR="${1:?Usage: sbatch scripts/e2e_routing_analysis.sh <save_dir>}"
CHECKPOINT_DIR=$(ls -d "$SAVE_DIR"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: No checkpoint found in $SAVE_DIR"
    exit 1
fi

echo "=== Step 1: Extract ECI routing data ==="
echo "Checkpoint: $CHECKPOINT_DIR"

if [ -f "${CHECKPOINT_DIR}/eci_routing_data.pt" ]; then
    echo "Found existing ${CHECKPOINT_DIR}/eci_routing_data.pt, skipping extraction"
else
    python scripts/extract_routing_vectors.py \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --dataset-dir data/nemotron-moe-exam \
        --max-len 2048 \
        --batch-size 4 \
        --mode eci \
        --num-viz-samples 20 \
        --num-gpus 4 \
        --output "${CHECKPOINT_DIR}/eci_routing_data.pt"

    if [ ! -f "${CHECKPOINT_DIR}/eci_routing_data.pt" ]; then
        echo "ERROR: Extraction failed"
        exit 1
    fi
fi

echo "=== Step 2: Compute ECI metrics and plots ==="

python scripts/compute_eci_metrics.py \
    --input "${CHECKPOINT_DIR}/eci_routing_data.pt" \
    --output-dir "${CHECKPOINT_DIR}/eci_results"

echo "=== Step 3: Gate weight PCA & similarity analysis ==="

python scripts/analyze_gate_weights.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "${CHECKPOINT_DIR}/gate_analysis"

echo "=== Done ==="
