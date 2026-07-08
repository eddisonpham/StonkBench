#!/bin/bash
#SBATCH --job-name=sb-hp
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/hp_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/hp_%A_%a.err

set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

TRIALS_PER_TASK=9
START=$((SLURM_ARRAY_TASK_ID * TRIALS_PER_TASK))
END=$((START + TRIALS_PER_TASK - 1))
PARALLEL_JOBS="${HP_PARALLEL:-6}"

run_trial() {
  local trial_id="$1"
  python -m src.experiments.hp_search \
    --device "${STONKBENCH_DEVICE:-cuda}" \
    --output_dir "${OUTPUT_ROOT}/results/hp_search" \
    --trial_id "${trial_id}"
}

export -f run_trial
export OUTPUT_ROOT PROJECT_ROOT STONKBENCH_DEVICE

seq "${START}" "${END}" | parallel -j "${PARALLEL_JOBS}" run_trial {}
