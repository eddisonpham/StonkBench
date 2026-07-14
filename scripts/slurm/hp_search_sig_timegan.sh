#!/bin/bash
#SBATCH --job-name=sb-hp-sig
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-26%8
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/hp_sig_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/hp_sig_%A_%a.err

# Targeted HP search: timegan + pcf_gan + sig_wgan (3 x 9 = 27 trials).
# Must pass --models so trial_id 0..26 maps to this subset only.
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
  --models timegan pcf_gan sig_wgan \
  "${SMOKE_ARGS[@]}"
