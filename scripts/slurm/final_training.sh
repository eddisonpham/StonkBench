#!/bin/bash
#SBATCH --job-name=sb-train
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/train_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/train_%A_%a.err

set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

MODELS=(
  quantgan
  timegan
  timegrad
  timevae
  unconditional_tsdiffusion
  vrnn
  gbm_adapter
  block_bootstrap
  ou_process
  merton_jump_diffusion
  de_jump_diffusion
  garch11
)

MODELS_PER_TASK=3
START=$((SLURM_ARRAY_TASK_ID * MODELS_PER_TASK))
END=$((START + MODELS_PER_TASK - 1))
PARALLEL_JOBS="${TRAIN_PARALLEL:-3}"

for idx in $(seq "${START}" "${END}"); do
  python -m src.experiments.run_final_training \
    --hp_summary "${OUTPUT_ROOT}/results/hp_search/summary.json" \
    --output_root "${OUTPUT_ROOT}" \
    --generation_length "${GENERATION_LENGTH:-21}" \
    --num_samples "${NUM_SAMPLES:-1000}" \
    --seed "${SEED:-42}" \
    --device "${STONKBENCH_DEVICE:-cuda}" \
    --models "${MODELS[$idx]}" &
done
wait
