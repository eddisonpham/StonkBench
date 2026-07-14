#!/bin/bash
#SBATCH --job-name=sb-train-sig
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/train_sig_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/train_sig_%A_%a.err

# Final DL training for signature/TimeGAN subset only.
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

MODELS=(
  timegan
  pcf_gan
  sig_wgan
)

MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"

SMOKE_ARGS=()
if [[ "${STONKBENCH_SMOKE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--smoke)
fi

echo "=== DL TRAIN ${MODEL} run=${STONKBENCH_RUN_ID} gen_len=${GENERATION_LENGTH:-100} $(date -Is) ==="
python -m src.experiments.run_final_training \
  --hp_summary "${RESULTS_DIR}/hp_search/summary.json" \
  --output_root "${OUTPUT_ROOT}" \
  --run_id "${STONKBENCH_RUN_ID}" \
  --generation_length "${GENERATION_LENGTH:-100}" \
  --num_samples "${NUM_SAMPLES:-1000}" \
  --seed "${SEED:-42}" \
  --device "${STONKBENCH_DEVICE:-cuda}" \
  --models "${MODEL}" \
  "${SMOKE_ARGS[@]}"
echo "=== DONE ${MODEL} $(date -Is) ==="
