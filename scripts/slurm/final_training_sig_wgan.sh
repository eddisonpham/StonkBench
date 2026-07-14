#!/bin/bash
#SBATCH --job-name=sb-sigwgan
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/train_sigwgan_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/train_sigwgan_%j.err

# Final training for sig_wgan only (path-space + vendor augs fix).
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

HP_SUMMARY="${HP_SUMMARY:-${RESULTS_DIR}/hp_search/summary.json}"
if [[ ! -f "${HP_SUMMARY}" ]]; then
  # Allow pointing a new RUN_ID at a prior HP summary.
  HP_SUMMARY="${SCRATCH_ROOT}/stonkbench/output/results/2026-07-12_sig_timegan/hp_search/summary.json"
fi

SMOKE_ARGS=()
if [[ "${STONKBENCH_SMOKE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--smoke)
fi

echo "=== DL TRAIN sig_wgan run=${STONKBENCH_RUN_ID} gen_len=${GENERATION_LENGTH:-100} $(date -Is) ==="
echo "HP_SUMMARY=${HP_SUMMARY}"
python -m src.experiments.run_final_training \
  --hp_summary "${HP_SUMMARY}" \
  --output_root "${OUTPUT_ROOT}" \
  --run_id "${STONKBENCH_RUN_ID}" \
  --generation_length "${GENERATION_LENGTH:-100}" \
  --num_samples "${NUM_SAMPLES:-1000}" \
  --seed "${SEED:-42}" \
  --device "${STONKBENCH_DEVICE:-cuda}" \
  --models sig_wgan \
  "${SMOKE_ARGS[@]}"
echo "=== DONE sig_wgan $(date -Is) ==="
