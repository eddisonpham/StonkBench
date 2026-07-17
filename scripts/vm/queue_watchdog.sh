#!/usr/bin/env bash
# scripts/vm/queue_watchdog.sh
#
# Continuous pipeline watchdog for StonkBench. Designed to run in parallel
# with scripts/vm/_submit_run.sh and surface queue health that the
# orchestrator's spot-check logs don't reveal:
#
#   - Are all N_HP_TRIALS=72 trials actually launched (vs. silently skipped)?
#   - Is the GPU-utilization curve consistent with the cap (no under-util)?
#   - Did the orchestrator die while HP search was in flight?
#   - Did we hit a fresh FATAL/OOM after the last successful completion?
#
# Usage:
#   nohup bash scripts/vm/queue_watchdog.sh &
#   # stdout/stderr streamed to /tmp/stonkbench-logs/queue_watchdog.log
#   tail -f /tmp/stonkbench-logs/queue_watchdog.log
#
# Behavioral contract:
#   - Snapshots every --interval seconds (default 60).
#   - Emits one line per snapshot with the four metrics.
#   - Emits an ALERT line when any of these trigger:
#       (a) Orchestrator + children both dead but trials remain.
#       (b) No new trial JSON for >= STALE_THRESHOLD seconds while at least
#           one child is alive (likely OOM-rotation / cap too tight).
#       (c) Recent FATAL / CUDA-OOM in the orchestrator log.
#   - Exits cleanly (rc 0) when 72 trials are all complete OR on SIGTERM.
#
# Env vars (positional RUN_ID overrides env-overrides-default-latest-dir):
#   STONKBENCH_RUN_ID  the run to monitor (defaults to latest dated subdir).
#   STONKBENCH_OUTPUT_ROOT  (defaults to $PROJECT_ROOT/outputs).
#   INTERVAL  seconds between snapshots (default 60).
#   STALE_THRESHOLD  seconds of no-progress-while-procs-alive (default 600).

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/phamnhut/StonkBench}"
OUTPUT_ROOT="${STONKBENCH_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
LOG_DIR_DEFAULT="${OUTPUT_ROOT}/logs"
WATCHDOG_LOG_DIR="${LOG_DIR_DEFAULT}"

# Resolve RUN_ID: positional arg > env > latest dated subdir of $OUTPUT_ROOT/results.
RUN_ID="${1:-${STONKBENCH_RUN_ID:-}}"
if [[ -z "${RUN_ID}" && -d "${OUTPUT_ROOT}/results" ]]; then
    RUN_ID="$(ls -1 "${OUTPUT_ROOT}/results" 2>/dev/null \
        | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -r | head -1 || true)"
fi
if [[ -z "${RUN_ID}" ]]; then
    echo "[queue_watchdog] ERROR: cannot resolve RUN_ID (no positional arg, env, or results dir)" >&2
    exit 1
fi

INTERVAL="${INTERVAL:-60}"
STALE_THRESHOLD="${STALE_THRESHOLD:-600}"   # 10 min default

HP_DIR="${OUTPUT_ROOT}/results/${RUN_ID}/hp_search"
TRIALS_DIR="${HP_DIR}/trials"
LOG_DIR="${OUTPUT_ROOT}/logs/${RUN_ID}/hp_search"
ORCH_LOG="/tmp/stonkbench-logs/stonkbench_run_${RUN_ID}.log"
WATCHDOG_LOG="${WATCHDOG_LOG_DIR}/queue_watchdog_${RUN_ID}.log"

mkdir -p "${WATCHDOG_LOG_DIR}"

# Pre-compute N_HP_TRIALS from hp_configs.py (same derivation run_parallel.sh uses).
N_HP_TRIALS=$(PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python - <<'PY' 2>/dev/null || echo 72
from src.experiments.hp_configs import DL_MODEL_KEYS, configs_for_model
print(sum(len(configs_for_model(m)) for m in DL_MODEL_KEYS))
PY
)
if ! [[ "${N_HP_TRIALS}" =~ ^[0-9]+$ ]] || [[ "${N_HP_TRIALS}" -lt 1 ]]; then
    N_HP_TRIALS=72
fi

log() {
    # Always append so a `nohup`-style launch persists across disconnects.
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "${WATCHDOG_LOG}"
}

# Cleanup trap: SIGTERM/SIGINT prints a final summary, then exits clean.
cleanup() {
    log "WATCHDOG exiting cleanly (sig received or pipeline done)."
    exit 0
}
trap cleanup SIGTERM SIGINT

log "=== queue_watchdog started ==="
log "  RUN_ID            = ${RUN_ID}"
log "  OUTPUT_ROOT       = ${OUTPUT_ROOT}"
log "  HP_DIR            = ${HP_DIR}"
log "  ORCH_LOG          = ${ORCH_LOG}"
log "  WATCHDOG_LOG      = ${WATCHDOG_LOG}"
log "  N_HP_TRIALS       = ${N_HP_TRIALS}"
log "  INTERVAL          = ${INTERVAL}s"
log "  STALE_THRESHOLD   = ${STALE_THRESHOLD}s"

# State carried across iterations.
prev_count=-1
stale_seconds=0
last_alert_signature=""

emit_alert() {
    local level="$1"; local msg="$2"
    # Dedup on `level` only (state-change semantics). Different ALERTs come
    # from different `if` branches in the health checks, so a level change
    # IS a state change (DEAD→STALLED, etc.) and re-emits. Same level means
    # "still in this state" → suppress to avoid spam. The per-iteration
    # snapshot line below keeps the live metrics visible at all times.
    # NOTE: previously keyed on `${level}:${msg}`, but `msg` embeds dynamic
    # numbers (remaining count, stale_seconds, alive procs) that change
    # every minute even while the condition holds — that signature-based
    # dedup effectively never fired, so DEAD/STALLED alerts emitted every
    # iteration instead of on the first detection.
    if [[ "${level}" == "${last_alert_signature}" ]]; then
        return
    fi
    last_alert_signature="${level}"
    log "ALERT[${level}]: ${msg}"
}

# Main loop.
while true; do
    # 1) Read current state.
    completed=0
    [[ -d "${TRIALS_DIR}" ]] && completed=$(ls -1 "${TRIALS_DIR}"/*.json 2>/dev/null | wc -l)
    alive=$(pgrep -af 'src.experiments.hp_search' 2>/dev/null | wc -l)
    orch_alive=$(pgrep -af 'run_parallel' 2>/dev/null | wc -l)
    gpu_free='?'
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_free=$(timeout 5 nvidia-smi --query-gpu=memory.free \
            --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \t')
        [[ -z "${gpu_free}" || ! "${gpu_free}" =~ ^[0-9]+$ ]] && gpu_free='?'
    fi
    remaining=$(( N_HP_TRIALS - completed ))

    # 2) Emit snapshot.
    log "snapshot: trials=${completed}/${N_HP_TRIALS} remaining=${remaining} alive_procs=${alive} orch_alive=${orch_alive} gpu_free_mib=${gpu_free}"

    # 3) Health checks.
    # 3a) Pipeline dead: orchestrator gone AND no children remaining > 0.
    if [[ ${remaining} -gt 0 && ${orch_alive} -eq 0 && ${alive} -eq 0 ]]; then
        emit_alert "CRITICAL" \
            "pipeline DEAD: orchestrator + children both gone while ${remaining} trial(s) remain. Run: tail -50 ${ORCH_LOG} ; then pkill -f 'python -m src.experiments.hp_search' ; bash scripts/vm/_submit_run.sh"
    fi
    # 3b) Stalled: procs alive but no progress for STALE_THRESHOLD seconds.
    if [[ ${completed} -eq ${prev_count} ]]; then
        stale_seconds=$(( stale_seconds + INTERVAL ))
    else
        stale_seconds=0
    fi
    if [[ ${stale_seconds} -ge ${STALE_THRESHOLD} && ${alive} -gt 0 ]]; then
        emit_alert "STALLED" \
            "no new JSON in ${stale_seconds}s while ${alive} child(ren) alive. Check ${LOG_DIR}/trial_*.log for OOMs."
    fi
    if [[ ${stale_seconds} -ge ${STALE_THRESHOLD} && ${alive} -eq 0 && ${orch_alive} -gt 0 ]]; then
        emit_alert "STALLED_ORCH" \
            "orchestrator alive but ${alive} children alive and no new JSON in ${stale_seconds}s. Likely child launch is failing."
    fi
    # 3c) Recent FATAL / OOM in orchestrator log (last 400 lines).
    if [[ -f "${ORCH_LOG}" ]]; then
        if tail -400 "${ORCH_LOG}" 2>/dev/null | grep -qE 'FATAL|OutOfMemory|CUDA out of memory'; then
            emit_alert "FATAL_DETECTED" \
                "recent FATAL or CUDA OOM in ${ORCH_LOG}; do NOT wait — investigate or restart."
        fi
    fi
    prev_count=${completed}

    # 4) Early exit when pipeline complete.
    if [[ ${remaining} -le 0 ]]; then
        log "all ${N_HP_TRIALS} trials complete; watchdog exiting."
        break
    fi

    # 5) Sleep, but break into smaller slices so SIGTERM is responsive.
    slept=0
    while (( slept < INTERVAL )); do
        sleep 5
        slept=$(( slept + 5 ))
    done
done

log "=== queue_watchdog done (rc=0) ==="
exit 0
