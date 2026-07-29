#!/bin/bash
#SBATCH --job-name=sb-train
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-12%4
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/train_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/train_%A_%a.err

# One model per GPU. 13 models total (7 DL + 6 statistical).
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

MODELS=(
  quantgan
  timegrad
  kalman_vae
  unconditional_tsdiffusion
  vrnn
  pcf_gan
  cond_sig_wgan
 
  block_bootstrap
 
  merton_jump_diffusion
  de_jump_diffusion
  garch11
)

MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"

SMOKE_ARGS=()
if [[ "${STONKBENCH_SMOKE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--smoke)
fi

python -m src.experiments.run_final_training \
  --hp_summary "${OUTPUT_ROOT}/results/hp_search/summary.json" \
  --output_root "${OUTPUT_ROOT}" \
  --generation_length "${GENERATION_LENGTH:-100}" \
  --num_samples "${NUM_SAMPLES:-1000}" \
  --seed "${SEED:-42}" \
  --device "${STONKBENCH_DEVICE:-cuda}" \
  --models "${MODEL}" \
  "${SMOKE_ARGS[@]}"
