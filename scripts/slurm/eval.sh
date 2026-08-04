#!/bin/bash
#SBATCH --job-name=sb-eval
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/eval_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/eval_%j.err

set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

python src/unified_evaluator.py \
  --generated_dir "${OUTPUT_ROOT}/results" \
  --results_dir "${OUTPUT_ROOT}/results/evaluation" \
  --seq_lengths ${SEQ_LENGTHS:-100}
