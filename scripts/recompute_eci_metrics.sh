#!/bin/bash
#SBATCH --job-name=eci_metrics
#SBATCH --output=eci_metrics_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=1:00:00

source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

echo "=== Temporal ==="
python scripts/compute_eci_metrics.py \
    --input checkpoints/temporal_9730558/step_30000/eci_routing_data.pt \
    --output-dir checkpoints/temporal_9730558/step_30000/eci_results

echo "=== Vanilla ==="
python scripts/compute_eci_metrics.py \
    --input checkpoints/vanilla_9730559/step_30000/eci_routing_data.pt \
    --output-dir checkpoints/vanilla_9730559/step_30000/eci_results

echo "=== Done ==="
