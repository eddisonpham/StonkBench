#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
CANONICAL_OUTPUT="/home/epham/StonkBench/output"
SCRATCH_ROOT="${SCRATCH:-/scratch/$USER}"
STAGING_ROOT="${SCRATCH_ROOT}/stonkbench/output"
if [[ "${STONKBENCH_SMOKE:-0}" == "1" ]]; then
  STAGING_ROOT="${SCRATCH_ROOT}/stonkbench/output_smoke"
fi
SLURM_LOG_DIR="${SCRATCH_ROOT}/stonkbench/slurm_logs"

# Dated run folder: everything under results/<RUN_ID>/
STONKBENCH_RUN_ID="${STONKBENCH_RUN_ID:-$(date +%Y-%m-%d)}"
export STONKBENCH_RUN_ID

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  OUTPUT_ROOT="${STAGING_ROOT}"
else
  OUTPUT_ROOT="${CANONICAL_OUTPUT}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export STONKBENCH_OUTPUT_ROOT="${OUTPUT_ROOT}"
export STONKBENCH_DL_SET_PATH="${PROJECT_ROOT}/data/preprocessed/dl_set.pt"
export STONKBENCH_STATS_SET_PATH="${PROJECT_ROOT}/data/preprocessed/statsmodel_set.pt"

RESULTS_DIR="${OUTPUT_ROOT}/results/${STONKBENCH_RUN_ID}"

if command -v module >/dev/null 2>&1; then
  module load StdEnv/2023 2>/dev/null || true
  # Trillium GPU: CUDA toolkit + sparse/DNN libs for official torch+cu126 wheels.
  module load gcc/12.3 cuda/12.6 2>/dev/null \
    || module load gcc/12.3 cuda/12.2 2>/dev/null \
    || true
  module load cudnn 2>/dev/null || true
  module load cusparselt 2>/dev/null || true
  module load nccl 2>/dev/null || true
fi

if [[ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda/etc/profile.d/conda.sh"
else
  # shellcheck source=/dev/null
  source "${HOME}/miniconda/bin/activate"
fi
conda activate stonkbench

# Alliance CUDA modules expose LIBRARY_PATH, not LD_LIBRARY_PATH. Torch needs the latter.
_cuda_lib_dirs=()
[[ -n "${EBROOTCUDA:-${CUDA_HOME:-}}" ]] && _cuda_lib_dirs+=(
  "${EBROOTCUDA:-$CUDA_HOME}/lib64"
  "${EBROOTCUDA:-$CUDA_HOME}/lib"
  "${EBROOTCUDA:-$CUDA_HOME}/extras/CUPTI/lib64"
)
[[ -n "${EBROOTCUDNN:-}" ]] && _cuda_lib_dirs+=("${EBROOTCUDNN}/lib")
[[ -n "${EBROOTCUSPARSELT:-}" ]] && _cuda_lib_dirs+=("${EBROOTCUSPARSELT}/lib")
[[ -n "${EBROOTNCCL:-}" ]] && _cuda_lib_dirs+=("${EBROOTNCCL}/lib")
_cuda_lib_dirs+=("${CONDA_PREFIX}/lib")
for _d in "${_cuda_lib_dirs[@]}"; do
  if [[ -d "${_d}" ]]; then
    export LD_LIBRARY_PATH="${_d}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
unset _cuda_lib_dirs _d

mkdir -p \
  "${RESULTS_DIR}" \
  "${OUTPUT_ROOT}/checkpoints/${STONKBENCH_RUN_ID}" \
  "${OUTPUT_ROOT}/experiments/${STONKBENCH_RUN_ID}" \
  "${OUTPUT_ROOT}/logs/${STONKBENCH_RUN_ID}" \
  "${OUTPUT_ROOT}/sanity/${STONKBENCH_RUN_ID}" \
  "${SLURM_LOG_DIR}"

echo "STONKBENCH_RUN_ID=${STONKBENCH_RUN_ID}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
