#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
# Canonical output sits at <repo>/outputs/ (matches the runtime default
# STONKBENCH_OUTPUT_ROOT in src/utils/env.py). Override via $STONKBENCH_OUTPUT_ROOT
# or the per-script --output_root flag if you must point elsewhere.
CANONICAL_OUTPUT="${STONKBENCH_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
SCRATCH_ROOT="${SCRATCH:-/scratch/$USER}"
STAGING_ROOT="${SCRATCH_ROOT}/stonkbench/output"
if [[ "${STONKBENCH_SMOKE:-0}" == "1" ]]; then
  STAGING_ROOT="${SCRATCH_ROOT}/stonkbench/output_smoke"
fi
SLURM_LOG_DIR="${SCRATCH_ROOT}/stonkbench/slurm_logs"

# Run id token. Default is "latest" so re-runs overwrite in-place;
# ``archive_existing`` evacuates the prior active contents to ``_legacy/``.
# Override for an isolated scope via $STONKBENCH_RUN_ID (e.g. date stamps,
# HP-search brackets, ablation tags).
STONKBENCH_RUN_ID="${STONKBENCH_RUN_ID:-latest}"
export STONKBENCH_RUN_ID

# On compute nodes /home is typically read-only, so we write to scratch
# during computation and symlink ~/outputs -> scratch after completion.
# Detect compute vs login: if /home is not writable, fall back to scratch.
if mkdir -p "${CANONICAL_OUTPUT}/.write_test" 2>/dev/null; then
  rmdir "${CANONICAL_OUTPUT}/.write_test" 2>/dev/null || true
  OUTPUT_ROOT="${CANONICAL_OUTPUT}"
  _ON_COMPUTE=0
else
  OUTPUT_ROOT="${STAGING_ROOT}"
  _ON_COMPUTE=1
fi
export _ON_COMPUTE

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export STONKBENCH_OUTPUT_ROOT="${OUTPUT_ROOT}"
export STONKBENCH_DL_SET_PATH="${PROJECT_ROOT}/data/preprocessed/dl_set.pt"
export STONKBENCH_STATS_SET_PATH="${PROJECT_ROOT}/data/preprocessed/statsmodel_set.pt"

RESULTS_DIR="${OUTPUT_ROOT}/results/${STONKBENCH_RUN_ID}"

if command -v module >/dev/null 2>&1; then
  module load StdEnv/2023 2>/dev/null || true
  # cu130 torch wheel + gluonts expect libze_loader.so.1 (Intel Level-Zero).
  # Hoisted FIRST so it's visible to any subsequent cuda/cudnn modulefile that
  # may transitively depend on it (Trillium `cuda/12.6` does, but other
  # partitions' cuda modules might not). On partitions without this module the
  # WARN lands in the .err file so the operator sees it (stdout is unaffected).
  module load level-zero 2>/dev/null || \
    echo "[common.sh] WARN: level-zero unavailable; torch may fail libze_loader lookup at runtime" >&2
  # Trillium GPU: CUDA toolkit + sparse/DNN libs for official torch+cu126 wheels.
  module load gcc/12.3 cuda/12.6 2>/dev/null \
    || module load gcc/12.3 cuda/12.2 2>/dev/null \
    || true
  # Always load system cuDNN. PyTorch wheels from pytorch.org (>=2.5) and the
  # compute-canada torch 2.6+computecanada wheel do NOT bundle a libcudnn.so;
  # torch links in cudnn at runtime via ctypes/CUDA backend and rejects any
  # node-provided cuDNN whose version doesn't match its compiled ABI. The
  # Trillium compute partition's `cudnn` modulefile provides cuDNN 9.10
  # cuDNN that matches torch 2.6/cu12 compiled ABI; loading it onto
  # LD_LIBRARY_PATH is what's required.
  module load cudnn 2>/dev/null || true
  module load cusparselt 2>/dev/null || true
  module load nccl 2>/dev/null || true
fi

# Pre-flight: fail loud if neither env path is available. Saves debugging
# time on accounts/staging where the install hasn't been done.
if [[ ! -f "${HOME}/.venvs/stonkbench/bin/activate" ]] \
   && [[ ! -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]] \
   && [[ ! -f "${HOME}/miniconda/bin/activate" ]]; then
  echo "[common.sh] ERROR: no usable env (missing \$HOME/.venvs/stonkbench and \$HOME/miniconda)." >&2
  exit 2
fi

# Use venv (cu130 wheel from pytorch.org) instead of conda 'stonkbench'.
# The conda env doesn't exist on this account, AND the cu126 wheel pinned by
# scripts/install_stonkbench.sh wouldn't match the 13.0 driver anyway. The
# cu130 wheel matches the driver ABI; libze_loader.so.1 + other Level-Zero
# deps are still expected from the host (typically via `module load
# level-zero` or system ldconfig — calling modules below covers the common
# case; if a compute node diverges, add `module load level-zero` here).
if [[ -f "${HOME}/.venvs/stonkbench/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.venvs/stonkbench/bin/activate"
else
  # Legacy conda path for accounts that DO have conda-stonkbench.
  if [[ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda/etc/profile.d/conda.sh"
  else
    # shellcheck source=/dev/null
    source "${HOME}/miniconda/bin/activate"
  fi
  conda activate stonkbench
fi

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
# Optional conda-lib fallback. Guarded by `set -u`: if venv branch was taken,
# CONDA_PREFIX is unset and we must skip rather than append `/lib` to LD path.
[[ -n "${CONDA_PREFIX:-}" ]] && _cuda_lib_dirs+=("${CONDA_PREFIX}/lib")
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
