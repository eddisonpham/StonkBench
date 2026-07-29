#!/bin/bash
#SBATCH --job-name=stonkbench-neptune
#SBATCH --account=def-yqhuang
#SBATCH --nodes=1
#SBATCH --output=/scratch/%u/stonkbench/logs/neptune_benchmark_%j.out
#SBATCH --error=/scratch/%u/stonkbench/logs/neptune_benchmark_%j.err
#SBATCH --partition=compute_neptune
#SBATCH --qos=neptune
#SBATCH --nodes=1
#SBATCH --time=12:00:00

# Largest CCDB nodes: Neptune (160 logical cores, ~467 GiB RAM per node).
# Requires neptune QOS/allocation. Logs and artifacts go to $SCRATCH.

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/slurm/common.sh"

cd "${PROJECT_ROOT}"

GENERATION_LENGTH="${GENERATION_LENGTH:-52}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-15}"
MODELS="${MODELS:-quantgan block_bootstrap}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${JOB_ROOT}/experiments}"

python -m src.experiments.run_benchmark \
  --generation_length "${GENERATION_LENGTH}" \
  --num_samples "${NUM_SAMPLES}" \
  --num_epochs "${NUM_EPOCHS}" \
  --device cuda \
  --output_root "${OUTPUT_ROOT}" \
  --models ${MODELS}
