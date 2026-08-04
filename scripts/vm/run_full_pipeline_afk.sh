#!/usr/bin/env bash
# AFK-friendly end-to-end pipeline: trains all 14 models sequentially with
# vendor_best HP (per "NO HP TUNING" directive 2026-07-30). Auto-recovers from
# transient CUDA failures via 3-retry loop. Designed to be launched in a tmux
# session so the user can close the tab and come back tomorrow.
#
# Usage:
#   bash scripts/vm/run_full_pipeline_afk.sh                  # all 14 models on GPU
#   CPU_ONLY=1 bash scripts/vm/run_full_pipeline_afk.sh       # 6 stat models on CPU
#   STONKBENCH_DEVICE=cpu bash scripts/vm/run_full_pipeline_afk.sh   # force CPU
#
# Logging:
#   /tmp/full_pipeline_afk.log              — overall summary
#   /tmp/full_pipeline_afk.<model>.log      — per-model stdout/stderr
#   /tmp/full_pipeline_afk.done             — final summary (succeeded/failed/run_id)
#
# Env vars:
#   STONKBENCH_RUN_ID    — run id (default: <UTC date>_full_afk)
#   CPU_ONLY             — 1 to skip DL models (run only the 6 stat ones on CPU)
#   STONKBENCH_DEVICE    — override device (default: auto; CUDA if available else CPU)
#   GEN_LENGTH           — generation length (default: 252)
#   SEQ_LENGTHS          — additional seq lengths to trim (default: 21 42 126)
#   NUM_SAMPLES          — number of generated samples (default: 1000)
#   MAX_ATTEMPTS         — retries per model (default: 3)
#   COOLDOWN_SECONDS     — sleep between attempts (default: 30)

set -euo pipefail

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# shellcheck source=/dev/null
source "${HOME}/miniconda3/bin/activate" stonk

# Defaults
LOG_FILE="/tmp/full_pipeline_afk.log"
DONE_FILE="/tmp/full_pipeline_afk.done"
RUN_ID="${STONKBENCH_RUN_ID:-$(date -u +%Y-%m-%d)_full_afk}"
export STONKBENCH_RUN_ID="${RUN_ID}"
GEN_LENGTH="${GEN_LENGTH:-252}"
SEQ_LENGTHS="${SEQ_LENGTHS:-21 42 126}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-30}"
CPU_ONLY="${CPU_ONLY:-0}"
DEVICE="${STONKBENCH_DEVICE:-}"

# Setup output dirs
mkdir -p "outputs/results/${RUN_ID}" "outputs/sanity/${RUN_ID}" "outputs/checkpoints/${RUN_ID}"

# Helpers
log_ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log()    { printf '[%s] %s\n' "$(log_ts)" "$*" | tee -a "${LOG_FILE}"; }
err()    { printf '[%s] ERROR: %s\n' "$(log_ts)" "$*" | tee -a "${LOG_FILE}" >&2; }

log "============================================================"
log "StonkBench AFK pipeline starting"
log "project_root: ${PROJECT_ROOT}"
log "run_id:       ${RUN_ID}"
log "device:       ${DEVICE:-auto}"
log "cpu_only:     ${CPU_ONLY}"
log "git_commit:   $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
log "git_status:   $(git status --short 2>/dev/null | head -20 || echo 'unknown')"
log "torch:        $(python -c 'import torch; print(torch.__version__, "cuda=", torch.cuda.is_available())' 2>&1 | head -1)"
log "============================================================"

# Models — sourced from src/experiments/core/registry.py ADAPTER_REGISTRY.
# 8 DL + 5 stat = 13 total. (gbm_adapter and ou_process were removed from
# the pipeline earlier per user directive; stationary_block_bootstrap was
# added as a replacement block-bootstrap variant.)
DL_MODELS=(
    "quantgan"
    "vrnn"
    "kalman_vae"
    "unconditional_tsdiffusion"
    "conditional_tsdiffusion"
    "cond_sig_wgan"
    "timegrad"
)
STAT_MODELS=(
    "block_bootstrap"
    "stationary_block_bootstrap"
    "merton_jump_diffusion"
    "de_jump_diffusion"
    "garch11"
)

# GPU sanity check: real CUDA ops, not just torch import
gpu_alive() {
    python -c "
import torch
x = torch.randn(100, 100, device='cuda')
y = torch.randn(100, 100, device='cuda')
z = (x @ y).sum().item()
torch.cuda.synchronize()
print(f'CUDA OK: z={z:.4f}')
" 2>&1 | tee -a "${LOG_FILE}"
}

if [ "${CPU_ONLY}" -eq 1 ]; then
    MODELS=("${STAT_MODELS[@]}")
    DEVICE="cpu"
    log "[cpu_only] skipping DL models — running ${#STAT_MODELS[@]} statistical models on CPU"
elif [ -n "${DEVICE}" ]; then
    MODELS=("${DL_MODELS[@]}" "${STAT_MODELS[@]}")
    log "[explicit-device] using device=${DEVICE} for all ${#MODELS[@]} models"
else
    log "[gpu-sanity] running real CUDA ops check before launching DL models..."
    if ! gpu_alive | grep -q "CUDA OK"; then
        err "GPU is broken (CUDA error). Options:"
        err "  (a) Reboot the VM, then re-run: bash ${SCRIPT_DIR}/run_full_pipeline_afk.sh"
        err "  (b) Run stat models only on CPU: CPU_ONLY=1 bash ${SCRIPT_DIR}/run_full_pipeline_afk.sh"
        err "ABORTING"
        echo "status=aborted" > "${DONE_FILE}"
        echo "reason=gpu-broken" >> "${DONE_FILE}"
        exit 1
    fi
    log "✓ GPU sanity passed"
    MODELS=("${DL_MODELS[@]}" "${STAT_MODELS[@]}")
fi

# Run loop
failed_models=()
succeeded_models=()
skipped_models=()

for MODEL in "${MODELS[@]}"; do
    log "============================================================"
    log ">>> Training: ${MODEL}"
    log "============================================================"

    # Resume-by-skip: if the artifact already exists, skip the model
    ARTIFACT="outputs/results/${RUN_ID}/${MODEL}/artifacts/${MODEL}_seq${GEN_LENGTH}.pt"
    if [ -f "${ARTIFACT}" ]; then
        log "[${MODEL}] artifact exists at ${ARTIFACT}, skipping (resume-by-skip)"
        skipped_models+=("${MODEL}")
        continue
    fi

    success=0
    for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
        MODEL_LOG="/tmp/full_pipeline_afk.${MODEL}.log"
        log "[${MODEL}] attempt ${attempt}/${MAX_ATTEMPTS} — log: ${MODEL_LOG}"

        # Run python in its own process group so we can kill the whole subtree on failure
        set +e
        set -m  # enable job control for process group
        DEVICE_FLAG=""
        if [ -n "${DEVICE}" ]; then
            DEVICE_FLAG="--device ${DEVICE}"
        fi
        # shellcheck disable=SC2086
        python -m src.experiments.run_final_training \
            --models "${MODEL}" \
            --generation_length "${GEN_LENGTH}" \
            --seq_lengths ${SEQ_LENGTHS} \
            --num_samples "${NUM_SAMPLES}" \
            --seed 42 \
            --run_id "${RUN_ID}" \
            ${DEVICE_FLAG} \
            > "${MODEL_LOG}" 2>&1 &
        PYTHON_PID=$!
        PYTHON_PGID=$(ps -o pgid= "${PYTHON_PID}" 2>/dev/null | tr -d ' ')
        wait "${PYTHON_PID}"
        rc=$?
        # Verify artifact exists (catches silent failures from tee/redirect)
        set +e
        if [ "${rc}" -eq 0 ] && [ -f "${ARTIFACT}" ]; then
            log "[${MODEL}] attempt ${attempt} SUCCEEDED (artifact at ${ARTIFACT})"
            success=1
            break
        else
            err "[${MODEL}] attempt ${attempt} FAILED (rc=${rc}, artifact exists=$( [ -f "${ARTIFACT}" ] && echo yes || echo no ))"
            tail -15 "${MODEL_LOG}" | tee -a "${LOG_FILE}"
            if [ -n "${PYTHON_PGID}" ]; then
                log "[${MODEL}] killing process group ${PYTHON_PGID} + waiting ${COOLDOWN_SECONDS}s"
                kill -9 "-${PYTHON_PGID}" 2>/dev/null || true
            fi
            sleep "${COOLDOWN_SECONDS}"
        fi
    done

    if [ "${success}" -eq 1 ]; then
        succeeded_models+=("${MODEL}")
    else
        err "[${MODEL}] ALL ${MAX_ATTEMPTS} ATTEMPTS FAILED — moving to next"
        failed_models+=("${MODEL}")
    fi
done

# Summary
log "============================================================"
log "StonkBench AFK pipeline DONE"
log "succeeded: ${succeeded_models[*]:-none}"
log "skipped:   ${skipped_models[*]:-none}"
log "failed:    ${failed_models[*]:-none}"
log "Run id:    ${RUN_ID}"
log "Artifacts: outputs/results/${RUN_ID}/"
log "Sanity:    outputs/sanity/${RUN_ID}/"
log "============================================================"

{
    echo "status=done"
    echo "run_id=${RUN_ID}"
    echo "succeeded=${succeeded_models[*]:-none}"
    echo "skipped=${skipped_models[*]:-none}"
    echo "failed=${failed_models[*]:-none}"
    echo "completed_at=$(log_ts)"
} > "${DONE_FILE}"
