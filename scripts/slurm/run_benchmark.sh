#!/bin/bash
#SBATCH --job-name=stonkbench-gen
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/benchmark_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/benchmark_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/slurm/common.sh"

GENERATION_LENGTH="${GENERATION_LENGTH:-52}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-15}"
SEED="${SEED:-42}"
MODELS="${MODELS:-quantgan timegrad kalman_vae unconditional_tsdiffusion vrnn pcf_gan cond_sig_wgan block_bootstrap merton_jump_diffusion de_jump_diffusion garch11}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STONKBENCH_OUTPUT_ROOT:-/home/epham/StonkBench/output}}"
DEVICE="${STONKBENCH_DEVICE:-cuda}"

cd "${PROJECT_ROOT}"

python -m src.experiments.run_benchmark \
  --generation_length "${GENERATION_LENGTH}" \
  --num_samples "${NUM_SAMPLES}" \
  --num_epochs "${NUM_EPOCHS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --output_root "${OUTPUT_ROOT}" \
  --models ${MODELS}
