#!/bin/bash
#SBATCH --job-name=sb-bootstrap
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/bootstrap_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/bootstrap_%j.err

# Quick generation of missing block_bootstrap and stationary_block_bootstrap
# artifacts at seq_lengths 21, 42, 126, 252.
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

echo "=== Generating block_bootstrap (all 4 seq lengths) ==="
python -m src.experiments.run_final_training \
  --generation_length 252 --seq_lengths 21 42 126 \
  --num_samples 1000 --seed 42 --device cpu \
  --models block_bootstrap --output_root "${OUTPUT_ROOT}"

echo "=== Generating stationary_block_bootstrap (missing 21,42,126) ==="
python -m src.experiments.run_final_training \
  --generation_length 252 --seq_lengths 21 42 126 \
  --num_samples 1000 --seed 42 --device cpu \
  --models stationary_block_bootstrap --output_root "${OUTPUT_ROOT}"

echo "=== DONE ==="
