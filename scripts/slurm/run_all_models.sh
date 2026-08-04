#!/bin/bash
#SBATCH --job-name=sb-experiment
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/scratch/epham/stonkbench/slurm_logs/exp_%A_%a.out
#SBATCH --error=/scratch/epham/stonkbench/slurm_logs/exp_%A_%a.err

# Unified experiment: HP search + final train + generate per model.
# DL models get HP tuning; stat models get full-history training.
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"
mkdir -p /scratch/epham/stonkbench/slurm_logs

MODELS=(
  # --- DL models (7) ---
  quantgan
  timegrad
  kalman_vae
  unconditional_tsdiffusion
  conditional_tsdiffusion
  vrnn
  cond_sig_wgan
  # --- Statistical models (5) ---
  block_bootstrap
  stationary_block_bootstrap
  merton_jump_diffusion
  de_jump_diffusion
  garch11
)

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${#MODELS[@]}" ]]; then
  echo "[FATAL] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= ${#MODELS[@]}" >&2
  exit 2
fi

MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"

SMOKE_ARGS=()
if [[ "${STONKBENCH_SMOKE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--smoke)
fi

# Skip HP search for stat models (they have no grid); run HP for DL models.
HP_ARGS=()
# Statistical models have no HPConfig entries, so hp_search is a no-op for them.
# run_experiment.py handles this internally, but we pass --skip-hp for stat
# models to avoid unnecessary trial spawning.
case "${MODEL}" in
  block_bootstrap|stationary_block_bootstrap|merton_jump_diffusion|de_jump_diffusion|garch11)
    HP_ARGS+=(--skip-hp)
    ;;
esac

echo "=== EXPERIMENT ${MODEL} run=${STONKBENCH_RUN_ID} $(date -Is) ==="

python scripts/run_experiment.py \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${STONKBENCH_RUN_ID}" \
  --device "${STONKBENCH_DEVICE:-cuda}" \
  --models "${MODEL}" \
  "${HP_ARGS[@]}" \
  "${SMOKE_ARGS[@]}"

# If writing to scratch (compute node), print symlink hint for the user
if [[ "${_ON_COMPUTE:-0}" == "1" ]]; then
  echo "[run_all_models] Results written to: ${OUTPUT_ROOT}"
  echo "[run_all_models] After all jobs complete, run from login node:"
  echo "  ln -sfn ${OUTPUT_ROOT}/${STONKBENCH_RUN_ID} ${PROJECT_ROOT}/outputs/results/${STONKBENCH_RUN_ID}"
fi

echo "=== DONE ${MODEL} $(date -Is) ==="
