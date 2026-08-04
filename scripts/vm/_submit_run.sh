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

# RUN_ID resolution — date-locked at first submit, sticky across overnight restarts.
# The pipeline MUST consolidate into ONE dates/<latest>_run folder per logical
# session, even if a restart crosses midnight. The cookie lives at
# ${PROJECT_ROOT}/outputs/.active_run_id (NOT /tmp — survives reboots; covered
# by the project's `outputs/` gitignore). Cookie holds a BARE UTC date
# (e.g. "2026-07-17"); this script appends the "_run" suffix on consumption.
# Sharing with run_parallel.sh via scripts/vm/_run_id.sh, which appends "_vm".
# To FORCE a new run id (e.g. user wants to start a brand-new pipeline session),
# delete the cookie:
#   rm /home/phamnhut/StonkBench/outputs/.active_run_id
export STONKBENCH_RUN_ID_LOCK_FILE="${STONKBENCH_RUN_ID_LOCK_FILE:-${PROJECT_ROOT}/outputs/.active_run_id}"
export STONKBENCH_RUN_ID_LOCK_TTL_DAYS="${STONKBENCH_RUN_ID_LOCK_TTL_DAYS:-14}"

# Sourced AFTER PROJECT_ROOT is resolved (the helper uses $PROJECT_ROOT for
# the default lock path).
# shellcheck source=scripts/vm/_run_id.sh
source "${PROJECT_ROOT}/scripts/vm/_run_id.sh"

# One-time migration from pre-refactor cookie format. The previous cookie at
# /tmp/stonkbench_active_run_id held a full run id (e.g. "2026-07-17_run");
# the new cookie holds a bare date ("2026-07-17"). Idempotent — only fires
# when the new cookie is missing AND the legacy cookie exists with a known
# suffix; safe to re-run.
if [[ ! -f "${STONKBENCH_RUN_ID_LOCK_FILE}" && -f /tmp/stonkbench_active_run_id ]]; then
    legacy_val=$(tr -d '[:space:]' < /tmp/stonkbench_active_run_id)
    if [[ -n "${legacy_val}" ]]; then
        bare=$(printf '%s' "${legacy_val}" | sed -E 's/_(run|vm)$//')
        if [[ "${bare}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
            # Best-effort migration — guard against set -e abort by ignoring mkdir/printf failures.
            # Migration is signaling; the cookie will be re-written on the first fresh-date branch
            # inside sb_init_run_id anyway.
            mkdir -p "$(dirname "${STONKBENCH_RUN_ID_LOCK_FILE}")" 2>/dev/null || true
            printf '%s\n' "${bare}" > "${STONKBENCH_RUN_ID_LOCK_FILE}" 2>/dev/null || true
            if [[ -f "${STONKBENCH_RUN_ID_LOCK_FILE}" ]] && cmp -s <(printf '%s\n' "${bare}") "${STONKBENCH_RUN_ID_LOCK_FILE}"; then
                echo "[submit_run] migrated legacy cookie (/tmp/stonkbench_active_run_id: '${legacy_val}') -> ${STONKBENCH_RUN_ID_LOCK_FILE} (bare: '${bare}')" >&2
            fi
        fi
    fi
fi

# Honor explicit override OR resolve via cookie. sb_init_run_id sets
# STONKBENCH_RUN_ID in the caller scope.
: "${STONKBENCH_RUN_ID:=}"
sb_init_run_id "run"
export STONKBENCH_RUN_ID
LOG_FILE="${LOG_DIR}/stonkbench_run_${STONKBENCH_RUN_ID}.log"

# Per-stage concurrency cap. Defaults to 2 (Option A — proven safe for
# our 24 GiB GPU at L=252: two concurrent trials × ~6–8 GiB peak ≈ 14 GiB,
# safely under the 24 GiB limit even during PyTorch CUDA init spikes).
# Bumped down from 3 after the 2026-07-16 hp_search FATAL where 3
# concurrent allocs raced past 24 GiB and 43 of 46 trials OOMed. Override
# with `STONKBENCH_LOCAL_JOBS=N` in the calling shell if you want a
# different cap.
export STONKBENCH_LOCAL_JOBS="${STONKBENCH_LOCAL_JOBS:-2}"

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
