#!/bin/bash
#SBATCH --job-name=routing_analysis
#SBATCH --output=routing_analysis_%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:4
#SBATCH --partition=long
#SBATCH --time=3:00:00

# bash scripts/e2e_hf_analysis.sh deepseek-ai/deepseek-moe-16b-base results/eci_deepseek_ai
# bash scripts/e2e_hf_analysis.sh Qwen/Qwen1.5-MoE-A2.7B results/eci_qwen_1_5_moe

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1

MODEL_ID="${1:?Usage: sbatch scripts/e2e_hf_analysis.sh <hf_model_id> [output_dir]}"
OUTPUT_DIR="${2:-results/$(echo "$MODEL_ID" | tr '/' '_')}"
mkdir -p "$OUTPUT_DIR"

echo "=== Step 1: Extract ECI routing data ==="
echo "Model:  $MODEL_ID"
echo "Output: $OUTPUT_DIR"

if [ -f "${OUTPUT_DIR}/eci_routing_data.pt" ]; then
    echo "Found existing ${OUTPUT_DIR}/eci_routing_data.pt, skipping extraction"
else
    python scripts/extract_routing_vectors.py \
        --checkpoint-dir "$MODEL_ID" \
        --dataset-dir data/nemotron-moe-exam \
        --max-len 2048 \
        --batch-size 4 \
        --mode eci \
        --num-viz-samples 20 \
        --num-gpus 4 \
        --output-dir "$OUTPUT_DIR"

    if [ ! -f "${OUTPUT_DIR}/eci_routing_data.pt" ]; then
        echo "ERROR: Extraction failed"
        exit 1
    fi
fi

echo "=== Step 2: Compute ECI metrics and plots ==="

python scripts/compute_eci_metrics.py \
    --input "${OUTPUT_DIR}/eci_routing_data.pt" \
    --output-dir "${OUTPUT_DIR}/eci_results"

echo "=== Done ==="
