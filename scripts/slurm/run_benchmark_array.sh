#!/bin/bash
#SBATCH --job-name=stonkbench-array
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/benchmark_array_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/benchmark_array_%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/slurm/common.sh"

SEQ_LENGTHS=(52 60 120 180 240 300)
GENERATION_LENGTH="${SEQ_LENGTHS[$SLURM_ARRAY_TASK_ID]}"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-15}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STONKBENCH_OUTPUT_ROOT}}"
DEVICE="${STONKBENCH_DEVICE:-cuda}"

cd "${PROJECT_ROOT}"
echo "Array task ${SLURM_ARRAY_TASK_ID}: generation_length=${GENERATION_LENGTH}"

python -m src.experiments.run_benchmark \
  --generation_length "${GENERATION_LENGTH}" \
  --num_samples "${NUM_SAMPLES}" \
  --num_epochs "${NUM_EPOCHS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --output_root "${OUTPUT_ROOT}" \
  --models ${MODELS}
