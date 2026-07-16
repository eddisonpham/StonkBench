#!/usr/bin/env bash
# scripts/vm/_submit_run.sh
#
# Persistent launcher for the StonkBench end-to-end pipeline.
#
# Sets BOTH ``OUTPUT_ROOT`` (consumed by ``scripts/vm/run_parallel.sh``)
# AND ``STONKBENCH_OUTPUT_ROOT`` (consumed by ``src/utils/env.py`` and
# every other Python entrypoint). Without setting ``OUTPUT_ROOT`` too,
# ``run_parallel.sh`` defaults to ``/scratch/stonkbench/output`` which is
# unwritable on a vanilla VM and silently falls back to
# ``$HOME/stonkbench_output_vm`` — putting artifacts OUTSIDE the
# canonical ``<repo>/outputs/`` directory and violating the user's
# "everything under outputs/" constraint.
#
# Sources the user's miniconda, activates the ``stonk`` env, and execs
# ``run_parallel.sh all``. Pipes through ``tee`` to a /tmp logfile so the
# user can tail progress from a separate terminal without attaching to
# tmux.
#
# Usage:
#   bash scripts/vm/_submit_run.sh
#
# Idempotency: the parent invocation kills any pre-existing
# ``stonkbench_run`` tmux session first.
#
# Env vars (all optional, defaults shown):
#   STONKBENCH_OUTPUT_ROOT=/home/phamnhut/StonkBench/outputs
#   STONKBENCH_DL_SET_PATH=$PROJECT_ROOT/data/preprocessed/dl_set.pt
#   STONKBENCH_STATS_SET_PATH=$PROJECT_ROOT/data/preprocessed/statsmodel_set.pt
#   STONKBENCH_RUN_ID=$(date -u +%Y-%m-%d)_run
#   STONKBENCH_LOG_DIR=/tmp/stonkbench-logs

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/phamnhut/StonkBench}"
readonly PROJECT_ROOT

# Resolve a writable /tmp logfile path (under a per-user subdir so multiple
# users sharing the VM don't trample each other).
LOG_DIR="${STONKBENCH_LOG_DIR:-/tmp/stonkbench-logs}"
mkdir -p "${LOG_DIR}"

# Default run id to today's date so daily reruns land in distinct outputs/.
export STONKBENCH_RUN_ID="${STONKBENCH_RUN_ID:-$(date -u +%Y-%m-%d)_run}"
LOG_FILE="${LOG_DIR}/stonkbench_run_${STONKBENCH_RUN_ID}.log"

# === CRITICAL: set BOTH stamps so every consumer resolves to the same root. ===
# Capture any user-supplied STONKBENCH_OUTPUT_ROOT BEFORE we defensively
# unset, so a shell-profile export survives the tmux-server cache wipe.
# Without the OLD capture, an unset-then-default pattern silently discards
# the user's override. The unset itself kills any stale value inherited
# from a tmux-server cache (a prior failed run that fell back to
# $HOME/stonkbench_output_vm would otherwise persist across restart
# cycles, fragmenting artifacts across two trees).
OLD_ROOT="${STONKBENCH_OUTPUT_ROOT:-}"
unset OUTPUT_ROOT STONKBENCH_OUTPUT_ROOT 2>/dev/null || true
export STONKBENCH_OUTPUT_ROOT="${OLD_ROOT:-${PROJECT_ROOT}/outputs}"
export OUTPUT_ROOT="${STONKBENCH_OUTPUT_ROOT}"
export STONKBENCH_DL_SET_PATH="${STONKBENCH_DL_SET_PATH:-${PROJECT_ROOT}/data/preprocessed/dl_set.pt}"
export STONKBENCH_STATS_SET_PATH="${STONKBENCH_STATS_SET_PATH:-${PROJECT_ROOT}/data/preprocessed/statsmodel_set.pt}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[submit_run] PROJECT_ROOT=${PROJECT_ROOT}"            >&2
echo "[submit_run] STONKBENCH_RUN_ID=${STONKBENCH_RUN_ID}"  >&2
echo "[submit_run] STONKBENCH_OUTPUT_ROOT=${STONKBENCH_OUTPUT_ROOT}" >&2
echo "[submit_run] OUTPUT_ROOT=${OUTPUT_ROOT}"              >&2
echo "[submit_run] STONKBENCH_DL_SET_PATH=${STONKBENCH_DL_SET_PATH}"   >&2
echo "[submit_run] LOG_FILE=${LOG_FILE}" >&2

# Conda bootstrap. Try the standard init paths the user is likely to have.
CONDA_SH=""
for cand in \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniconda/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh" \
  ; do
  if [[ -f "${cand}" ]]; then
    CONDA_SH="${cand}"
    break
  fi
done
if [[ -n "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
else
  echo "[submit_run] WARN: conda.sh not found at standard locations; relying on direct PATH prepend" >&2
fi

# Activate the stonk env (idempotent). Fall back to direct PATH prepend if
# the conda CLI isn't reachable (e.g. conda isn't on $PATH after sourcing).
if ! conda activate stonk 2>/dev/null; then
  if [[ -x "/home/phamnhut/miniconda3/envs/stonk/bin/python" ]]; then
    export PATH="/home/phamnhut/miniconda3/envs/stonk/bin:${PATH}"
    export CONDA_PREFIX="/home/phamnhut/miniconda3/envs/stonk"
    echo "[submit_run] Activated stonk env via direct PATH prepend" >&2
  else
    echo "[submit_run] ERROR: cannot activate stonk env; aborting." >&2
    exit 1
  fi
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[submit_run] ERROR: python not on PATH after activation" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
echo "[submit_run] python:  $(python -V) ($(which python))"   >&2
echo "[submit_run] nvidia-smi: $(command -v nvidia-smi || echo MISSING)" >&2
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[submit_run] GPU count: $(nvidia-smi -L 2>/dev/null | wc -l)" >&2
fi
echo "[submit_run] Launch: bash scripts/vm/run_parallel.sh all (output piped to ${LOG_FILE})" >&2

# Stream every line to stderr (tmux scrollback) AND the persistent logfile
# so the user can monitor from a separate terminal. tee -a is safe across
# re-launches; the launcher is invoked once per pipeline so collisions
# are unlikely in practice.
exec bash scripts/vm/run_parallel.sh all 2>&1 | tee -a "${LOG_FILE}"
