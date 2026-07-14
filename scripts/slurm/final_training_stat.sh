#!/bin/bash
#SBATCH --job-name=sb-train-stat
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/train_stat_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/train_stat_%A_%a.err

# Statistical model fit + generate (no HP search). Generation length defaults to 100.
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

MODELS=(
  gbm_adapter
  block_bootstrap
  ou_process
  merton_jump_diffusion
  de_jump_diffusion
  garch11
)

MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"

SMOKE_ARGS=()
if [[ "${STONKBENCH_SMOKE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--smoke)
fi

HP_SUMMARY="${RESULTS_DIR}/hp_search/summary.json"
if [[ ! -f "${HP_SUMMARY}" ]]; then
  mkdir -p "$(dirname "${HP_SUMMARY}")"
  echo '{"models": {}}' > "${HP_SUMMARY}"
fi

echo "=== STAT FIT ${MODEL} run=${STONKBENCH_RUN_ID} gen_len=${GENERATION_LENGTH:-100} $(date -Is) ==="
python -m src.experiments.run_final_training \
  --hp_summary "${HP_SUMMARY}" \
  --output_root "${OUTPUT_ROOT}" \
  --run_id "${STONKBENCH_RUN_ID}" \
  --generation_length "${GENERATION_LENGTH:-100}" \
  --num_samples "${NUM_SAMPLES:-1000}" \
  --seed "${SEED:-42}" \
  --device "${STONKBENCH_DEVICE:-cuda}" \
  --models "${MODEL}" \
  "${SMOKE_ARGS[@]}"
echo "=== DONE ${MODEL} $(date -Is) ==="
