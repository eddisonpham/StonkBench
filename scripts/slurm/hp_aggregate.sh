#!/bin/bash
#SBATCH --job-name=sb-hp-agg
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/hp_agg_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/hp_agg_%j.err

set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

python -m src.experiments.hp_search \
  --aggregate_only \
  --output_dir "${OUTPUT_ROOT}/results/hp_search"
