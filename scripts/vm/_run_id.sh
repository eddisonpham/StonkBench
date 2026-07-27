#!/usr/bin/env bash
# scripts/vm/_run_id.sh
#
# Sticky RUN_ID resolution helper. Sourced by:
#   - scripts/vm/_submit_run.sh  (suffix="run")
#   - scripts/vm/run_parallel.sh (suffix="vm")
#
# Design (rationale lives at the top so future you doesn't re-invent it)
# ---------------------------------------------------------------------------
# Goal: every overnight restart must consolidate into ONE dates/<latest>_run
# folder per logical pipeline session — even when a restart crosses
# midnight UTC, even when a user invokes ``bash run_parallel.sh`` directly,
# even when a prior failed run left the cookie somewhere else.
#
# Cookie format:
#   * BARE UTC date, e.g. "2026-07-17". NO "_run" / "_vm" suffix in storage
#     — the suffix is per-consumer so each script can read the same cookie
#     and emit a different final run id without collision.
#   * Cookie is at ${STONKBENCH_RUN_ID_LOCK_FILE:-${PROJECT_ROOT}/outputs/.active_run_id}
#     (NOT /tmp — survives reboots; lives under outputs/ which is gitignored).
#
# Resolution order (first match wins):
#   1. Explicit env STONKBENCH_RUN_ID (highest priority, cookie untouched).
#   2. Cookie file present AND younger than STONKBENCH_RUN_ID_LOCK_TTL_DAYS
#      days (default 14). Bare date from cookie + caller-provided suffix.
#   3. Else: write today's UTC date to cookie, return bare+suffix.
#
# The bare-date / suffix split solves two problems simultaneously:
#   a. TZ drift: both scripts now read "2026-07-17" instead of disagreeing
#      on UTC vs local TZ at computing time.
#   b. Suffix drift: pre-refactor, _submit_run.sh wrote "_run" while
#      run_parallel.sh wrote "_vm" — whoever ran first "won" the suffix.
#      Now the cookie holds only the date, so each script appends its own
#      suffix deterministically.
#
# Usage:
#   source scripts/vm/_run_id.sh
#   sb_init_run_id "run"   # or "vm"
#   # $STONKBENCH_RUN_ID is now "2026-07-17_run" (or "2026-07-17_vm"); cookie updated/touched.
#
# Env inputs (read):
#   STONKBENCH_RUN_ID                  explicit override (highest priority)
#   STONKBENCH_RUN_ID_LOCK_FILE        default: ${PROJECT_ROOT}/outputs/.active_run_id
#   STONKBENCH_RUN_ID_LOCK_TTL_DAYS    default: 14
#   PROJECT_ROOT                       required for the default lock path resolution
#
# Env / caller-scope output (after sb_init_run_id returns 0):
#   STONKBENCH_RUN_ID                  resolved run id (bare_date + suffix)

# We intentionally do NOT put `set -euo pipefail` here — sourcing this file
# would re-enable strict mode in the caller's shell, which may already be
# under different (looser) flags. The caller is responsible for flags.

sb_init_run_id() {
    local suffix="${1:-run}"
    local lock="${STONKBENCH_RUN_ID_LOCK_FILE:-${PROJECT_ROOT:-/home/phamnhut/StonkBench}/outputs/.active_run_id}"
    local ttl="${STONKBENCH_RUN_ID_LOCK_TTL_DAYS:-14}"
    local now_epoch mtime_epoch age_days lock_val bare
    local fallback="/tmp/stonkbench_active_run_id"
    local migrated_from_tmpl=0

    # 1. Explicit env override wins.
    if [[ -n "${STONKBENCH_RUN_ID:-}" ]]; then
        echo "[run_id] STONKBENCH_RUN_ID=${STONKBENCH_RUN_ID} (explicit env override; cookie unchanged)" >&2
        return 0
    fi

    # 2. Cookie present, in-TTL, non-empty → reuse.
    if [[ -f "${lock}" ]]; then
        now_epoch=$(date +%s)
        mtime_epoch=$(stat -c %Y "${lock}" 2>/dev/null || echo 0)
        age_days=$(( (now_epoch - mtime_epoch) / 86400 ))
        if (( age_days < ttl )); then
            lock_val=$(tr -d '[:space:]' < "${lock}")
            if [[ -n "${lock_val}" ]]; then
                bare="${lock_val}"
                STONKBENCH_RUN_ID="${bare}_${suffix}"
                echo "[run_id] STONKBENCH_RUN_ID=${STONKBENCH_RUN_ID} (bare=${bare}; reused from cookie ${lock}; age=${age_days}d < ${ttl}d TTL)" >&2
                return 0
            fi
        fi
    fi

    # 3. Fresh: write today's UTC date to cookie.
    bare="$(date -u +%Y-%m-%d)"
    if ! { mkdir -p "$(dirname "${lock}")" && printf '%s\n' "${bare}" > "${lock}"; } 2>/dev/null; then
        # Cookie path unwritable — fall back to /tmp and WARN loudly. Cookie
        # gets written there so the in-process / same-shell-pipeline reuses
        # the same date, but a reboot would lose it. Only matters if the
        # caller has set STONKBENCH_RUN_ID_LOCK_FILE to something exotic.
        echo "[run_id] WARN: cannot write cookie to ${lock}; falling back to ${fallback}" >&2
        if ! printf '%s\n' "${bare}" > "${fallback}" 2>/dev/null; then
            echo "[run_id] ERROR: cannot write cookie to either ${lock} or ${fallback}; proceeding without persistent lock." >&2
            STONKBENCH_RUN_ID="${bare}_${suffix}"
            return 0
        fi
        STONKBENCH_RUN_ID_LOCK_FILE="${fallback}"
    fi
    STONKBENCH_RUN_ID="${bare}_${suffix}"
    echo "[run_id] STONKBENCH_RUN_ID=${STONKBENCH_RUN_ID} (bare=${bare}; fresh; cookie written)" >&2
    return 0
}

# Exported so consumers can `source` this and immediately call the function.
export -f sb_init_run_id
