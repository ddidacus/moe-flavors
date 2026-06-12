#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=64G
#SBATCH --time=4:00:00

# Run the job command passed as an argument when submitting the job ('python main.py' for example)
echo "Running command: $@"
srun uv run "$@"
