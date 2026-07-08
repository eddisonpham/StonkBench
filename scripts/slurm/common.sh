#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
CANONICAL_OUTPUT="/home/epham/StonkBench/output"
SCRATCH_ROOT="${SCRATCH:-/scratch/$USER}"
STAGING_ROOT="${SCRATCH_ROOT}/stonkbench/output"
SLURM_LOG_DIR="${SCRATCH_ROOT}/stonkbench/slurm_logs"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  OUTPUT_ROOT="${STAGING_ROOT}"
else
  OUTPUT_ROOT="${CANONICAL_OUTPUT}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export STONKBENCH_OUTPUT_ROOT="${OUTPUT_ROOT}"
export STONKBENCH_DL_SET_PATH="${PROJECT_ROOT}/data/preprocessed/dl_set.pt"
export STONKBENCH_STATS_SET_PATH="${PROJECT_ROOT}/data/preprocessed/statsmodel_set.pt"

if command -v module >/dev/null 2>&1; then
  module load StdEnv/2023 2>/dev/null || true
fi

if [[ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda/etc/profile.d/conda.sh"
else
  # shellcheck source=/dev/null
  source "${HOME}/miniconda/bin/activate"
fi
conda activate stonkbench
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUTPUT_ROOT}"/{checkpoints,experiments,logs,results,sanity} "${SLURM_LOG_DIR}"
