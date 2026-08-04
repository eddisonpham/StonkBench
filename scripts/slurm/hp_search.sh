#!/bin/bash
#SBATCH --job-name=sb-hp
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-71%8
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/hp_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/hp_%A_%a.err

# One HP trial per GPU. Full grid = 72 trials (8 DL models x 9 configs).
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

SMOKE_ARGS=()
if [[ "${STONKBENCH_SMOKE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--smoke)
fi

python -m src.experiments.hp_search \
  --device "${STONKBENCH_DEVICE:-cuda}" \
  --output_dir "${RESULTS_DIR}/hp_search" \
  --trial_id "${SLURM_ARRAY_TASK_ID}" \
  "${SMOKE_ARGS[@]}"
