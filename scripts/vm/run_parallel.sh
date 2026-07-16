#!/usr/bin/env bash
# scripts/vm/run_parallel.sh
#
# Single-stage orchestrator for running StonkBench on a multi-GPU VM without
# SLURM. Drives HP search, HP aggregation, final training, and unified
# evaluation as parallel background jobs, round-robin-pinned to detected
# NVIDIA GPUs by `scripts/vm/run_on_gpu.sh`.
#
# Stages (positional argument):
#   hp_search    - Launch all HP trials in parallel (count derived from
#                  src/experiments.hp_configs); aggregate when done.
#   aggregate    - Re-build summary.json from existing trial outputs (no GPUs).
#   final_train  - Train all 14 models (8 DL + 6 statistical) using HP winners.
#   eval         - Run unified_evaluator SERIALLY over artifacts under
#                  results/<RUN_ID>. Serial because every parallel instance
#                  would race on `<results_dir>/complete_evaluation.json`.
#   all          - hp_search -> final_train -> eval (sequential composition).
#
# Usage:
#   bash scripts/vm/run_parallel.sh all
#   bash scripts/vm/run_parallel.sh hp_search --gpu-count 4 --smoke
#   bash scripts/vm/run_parallel.sh final_train --run-id my_run --num-samples 1000
#   bash scripts/vm/run_parallel.sh eval --seq-lengths 100 180
#
# Env vars used (with defaults):
#   PROJECT_ROOT=$HOME/StonkBench
#   OUTPUT_ROOT=/scratch/stonkbench/output        (auto-fallback if not writable)
#   STONKBENCH_RUN_ID=$(date +%F)_vm              (override with --run-id)
#   STONKBENCH_DL_SET_PATH=$PROJECT_ROOT/data/preprocessed/dl_set.pt
#   STONKBENCH_STATS_SET_PATH=$PROJECT_ROOT/data/preprocessed/statsmodel_set.pt
#   N_GPUS=<auto-detected via nvidia-smi>          (override with --gpu-count)
#
# Notes:
# - This script does NOT call sbatch. It is for environments without SLURM.
# - GPU pinning via `CUDA_VISIBLE_DEVICES` per child process. See run_on_gpu.sh.
# - Logs land under `${OUTPUT_ROOT}/logs/${RUN_ID}/<phase>/<job>.log`.
# - Background-job failures propagate: `wait_for_jobs` exits non-zero on
#   any child failure (so a failed trial aborts the stage).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/vm"
RUN_ON_GPU="${SCRIPT_DIR}/run_on_gpu.sh"

# Defaults (env vars still override)
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/stonkbench/output}"
# Probe writability — fall back to a guaranteed-user-owned directory if the
# default path is not writable on this VM. This prevents broken `mkdir -p`
# in the orchestrator body when /scratch is not present. Avoid /tmp on
# multi-user VMs (world-readable); fall back to the current working
# directory if HOME is unset, which keeps artifacts under the user's
# launch shell.
if ! mkdir -p "${OUTPUT_ROOT}" 2>/dev/null; then
    FALLBACK="${HOME:-$(pwd)}/stonkbench_output_vm"
    echo "[run_parallel] WARN: cannot create OUTPUT_ROOT=${OUTPUT_ROOT}; using ${FALLBACK}" >&2
    OUTPUT_ROOT="${FALLBACK}"
    if ! mkdir -p "${OUTPUT_ROOT}" 2>/dev/null; then
        echo "[run_parallel] ERROR: cannot create fallback OUTPUT_ROOT=${OUTPUT_ROOT}" >&2
        echo "[run_parallel] pass --output-root or export OUTPUT_ROOT=/writable/path" >&2
        exit 1
    fi
fi
# Defense in depth: re-export OUTPUT_ROOT after the writability probe so any
# forked child (HP-search workers, final-train workers, eval) that re-reads
# OUTPUT_ROOT is guaranteed to see the canonical (probed) value, regardless
# of whether the invoking shell had OUTPUT_ROOT exported initially.
export OUTPUT_ROOT

# Per-stage concurrency cap. Each `launch()` blocks until the number of
# currently-alive children in BG_PIDS drops below this number. Prevents
# the classic 1-GPU OOM where launching all N_HP_TRIALS=72 HP trials at
# once races for a single ~24GB GPU. Set STONKBENCH_LOCAL_JOBS=0 to disable.
STONKBENCH_LOCAL_JOBS="${STONKBENCH_LOCAL_JOBS:-4}"
if ! [[ "${STONKBENCH_LOCAL_JOBS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "[run_parallel] WARN: STONKBENCH_LOCAL_JOBS=${STONKBENCH_LOCAL_JOBS} is not 0/a positive int; falling back to 4" >&2
    STONKBENCH_LOCAL_JOBS=4
fi
export STONKBENCH_LOCAL_JOBS
# Dynamic-cap control file. launch() reads this on every invocation, so
# external `echo N > ${STONKBENCH_LOCAL_JOBS_FILE}` updates take effect on
# the NEXT launch without restarting the orchestrator. Default location:
# /tmp/stonkbench_local_jobs. Set STONKBENCH_LOCAL_JOBS_FILE to override.
export STONKBENCH_LOCAL_JOBS_FILE="${STONKBENCH_LOCAL_JOBS_FILE:-/tmp/stonkbench_local_jobs}"
echo "${STONKBENCH_LOCAL_JOBS}" > "${STONKBENCH_LOCAL_JOBS_FILE}"

RUN_ID="${STONKBENCH_RUN_ID:-$(date +%F)_vm}"
# Default sequence length = 252 (≈1 trading year of daily bars). This bash
# default only flows into `stage_final_train` (`--generation_length`) and
# `stage_eval` (`--seq-lengths ${SEQ_LENGTHS}`). HP search ignores this env
# var and reads `int(dl_set["window_size"])` from the preprocessed tensors,
# so re-preprocess with `python -m src.data_preprocessing --window_size 252`
# before the first run if you intend L=252 throughout.
GENERATION_LENGTH="${GENERATION_LENGTH:-252}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
NUM_EPOCHS="${NUM_EPOCHS:-15}"
SEED="${SEED:-42}"
SMOKE=0
SEQ_LENGTHS=("${GENERATION_LENGTH}")

DL_SET_PATH="${STONKBENCH_DL_SET_PATH:-${PROJECT_ROOT}/data/preprocessed/dl_set.pt}"
STATS_SET_PATH="${STONKBENCH_STATS_SET_PATH:-${PROJECT_ROOT}/data/preprocessed/statsmodel_set.pt}"

# Background PID tracking — wait_for_jobs inspects these on drain.
BG_PIDS=()

# Accumulates rc values from children that died INSIDE the launch() gate
# (i.e., via `wait -n` rather than the per-PID `wait "${pid}"` drain at
# stage end). wait_for_jobs drains this at the start of each stage so a
# silent in-gate failure cannot pass the stage as healthy.
REAPED_FAILURES=()

MODELS_DL=(quantgan timegan timegrad timevae unconditional_tsdiffusion vrnn pcf_gan sig_wgan)
MODELS_STAT=(gbm_adapter block_bootstrap ou_process merton_jump_diffusion de_jump_diffusion garch11)
MODELS_ALL=("${MODELS_DL[@]}" "${MODELS_STAT[@]}")

# Number of HP trials (8 DL models × 9 configs = 72 today). Derived from
# hp_configs.py so adding/removing a model stays consistent.
N_HP_TRIALS="${N_HP_TRIALS:-}"
if [[ -z "${N_HP_TRIALS}" ]]; then
    N_HP_TRIALS=$(PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python - <<'PY' 2>/dev/null || echo 72
from src.experiments.hp_configs import DL_MODEL_KEYS, configs_for_model
print(sum(len(configs_for_model(m)) for m in DL_MODEL_KEYS))
PY
)
fi
if ! [[ "${N_HP_TRIALS}" =~ ^[0-9]+$ ]] || [[ "${N_HP_TRIALS}" -lt 1 ]]; then
    N_HP_TRIALS=72
fi

usage() {
    cat <<EOF
Usage: $(basename "$0") <stage> [options]

Stages (positional, required):
  hp_search   Run all ${N_HP_TRIALS} HP trials in parallel; aggregate on completion.
  aggregate   Rebuild summary.json from existing trial outputs (CPU).
  final_train Train all 14 models using HP winners.
  eval        Run unified_evaluator SERIALLY over artifacts under results/<RUN_ID>.
  all         hp_search -> final_train -> eval (sequential phases).

Options:
  --smoke                 HP search: pass --smoke; final_train: --smoke + ^num-samples=16.
  --gpu-count N           Override detected GPU count.
  --output-root PATH      Override \${OUTPUT_ROOT}.
  --run-id ID             Override \${STONKBENCH_RUN_ID}.
  --num-samples N         Default ${NUM_SAMPLES} (final_train / eval).
  --num-epochs N          Default ${NUM_EPOCHS}.
  --seq-lengths N [N ...] Sequence lengths to evaluate (default: ${GENERATION_LENGTH}).
  --help                  Print this message.

Env vars: PROJECT_ROOT, OUTPUT_ROOT, STONKBENCH_RUN_ID,
          STONKBENCH_DL_SET_PATH, STONKBENCH_STATS_SET_PATH, N_GPUS.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 0
fi

STAGE="$1"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1; shift ;;
        --gpu-count) N_GPUS="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --seq-lengths) shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SEQ_LENGTHS+=("$1"); shift; done ;;
        --help|-h) usage; exit 0 ;;
        *) echo "[run_parallel] unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Detect GPUs (best-effort). Honor N_GPUS override if set.
if [[ -z "${N_GPUS:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        DETECTED="$(nvidia-smi -L 2>/dev/null | wc -l)"
        if [[ "${DETECTED}" =~ ^[0-9]+$ ]] && [[ "${DETECTED}" -ge 1 ]]; then
            N_GPUS="${DETECTED}"
            echo "[run_parallel] detected ${N_GPUS} NVIDIA GPU(s)" >&2
        else
            echo "[run_parallel] WARN: nvidia-smi present but parses invalid (${DETECTED}); using N_GPUS=1" >&2
            N_GPUS=1
        fi
    else
        echo "[run_parallel] WARN: no nvidia-smi; assuming N_GPUS=1 (single pipeline, sequential)" >&2
        N_GPUS=1
    fi
fi
if ! [[ "${N_GPUS:-1}" =~ ^[0-9]+$ ]] || [[ "${N_GPUS:-1}" -lt 1 ]]; then N_GPUS=1; fi
# Round-robin uses this many slots — even on a 1-GPU VM, register N_GPUS so
# downstream tools (e.g. parallel -j ${N_GPUS}) see the correct cap.
export N_GPUS

# Effective num-samples/epochs in smoke
EFFECTIVE_NUM_SAMPLES="${NUM_SAMPLES}"
[[ "${SMOKE}" == "1" ]] && EFFECTIVE_NUM_SAMPLES=16

# Export so children pick them up.
export STONKBENCH_OUTPUT_ROOT="${OUTPUT_ROOT}"
export STONKBENCH_RUN_ID="${RUN_ID}"
export STONKBENCH_DL_SET_PATH="${DL_SET_PATH}"
export STONKBENCH_STATS_SET_PATH="${STATS_SET_PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[run_parallel] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[run_parallel] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[run_parallel] RUN_ID=${RUN_ID}"
echo "[run_parallel] N_GPUS=${N_GPUS}"
echo "[run_parallel] N_HP_TRIALS=${N_HP_TRIALS}"
echo "[run_parallel] SMOKE=${SMOKE}"
echo "[run_parallel] STAGE=${STAGE}"
echo "[run_parallel] STONKBENCH_LOCAL_JOBS=${STONKBENCH_LOCAL_JOBS} (per-stage concurrency cap; 0=unlimited)"

mkdir -p "${OUTPUT_ROOT}/logs/${RUN_ID}/hp_search" \
         "${OUTPUT_ROOT}/logs/${RUN_ID}/final_train" \
         "${OUTPUT_ROOT}/logs/${RUN_ID}/eval" \
         "${OUTPUT_ROOT}/results/${RUN_ID}" \
         "${OUTPUT_ROOT}/evaluation/${RUN_ID}"

# launch <slot> <log_path> -- command args...
launch() {
    local slot="$1"; local log="$2"; shift 2
    # Dynamic-cap read (control file). On every launch() invocation, peek at
    # STONKBENCH_LOCAL_JOBS_FILE so external `echo N > <file>` updates take
    # effect WITHOUT restarting the orchestrator. The control file is the
    # canonical channel for live cap changes; copying to it overrides the
    # env-set value until the next restart. Falls back silently if missing
    # or malformed so a transient file-state doesn't break the gate.
    if [[ -r "${STONKBENCH_LOCAL_JOBS_FILE:-/tmp/stonkbench_local_jobs}" ]]; then
        local file_cap
        file_cap=$(cat "${STONKBENCH_LOCAL_JOBS_FILE:-/tmp/stonkbench_local_jobs}" 2>/dev/null | tr -d '[:space:]')
        if [[ "${file_cap}" =~ ^(0|[1-9][0-9]*)$ ]] && [[ "${file_cap}" != "${STONKBENCH_LOCAL_JOBS}" ]]; then
            echo "[run_parallel] DYNAMIC CAP CHANGE: ${STONKBENCH_LOCAL_JOBS} -> ${file_cap} (read from ${STONKBENCH_LOCAL_JOBS_FILE})" >&2
            STONKBENCH_LOCAL_JOBS="${file_cap}"
        fi
    fi
    # GPU memory gate: refuse to launch a new trial until at least
    # STONKBENCH_GPU_GATE_MIB MiB of GPU memory is free. Prevents launching
    # a new DL trial on top of an in-flight TimeGAN-class trial (~22 GiB peak
    # at bs=128) that would OOM a 24 GB GPU. Fails open if nvidia-smi errors
    # (transient); sleeps 30s per retry, gives up after 60 retries (~30 min)
    # to avoid starving the pipeline forever on a hard GPU hold. Override
    # via env var; set to 0 to disable the gate entirely.
    local gpu_gate_mib="${STONKBENCH_GPU_GATE_MIB:-10000}"
    if [[ "${gpu_gate_mib}" =~ ^[0-9]+$ ]] && [[ "${gpu_gate_mib}" -gt 0 ]] \
        && command -v nvidia-smi >/dev/null 2>&1; then
        local probe_attempts=0
        while [[ ${probe_attempts} -lt 60 ]]; do
            local gpu_free probe_rc=0
            gpu_free=$(timeout 5 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) || probe_rc=$?
            if [[ ${probe_rc} -ne 0 || -z "${gpu_free}" || ! "${gpu_free}" =~ ^[0-9]+$ ]]; then
                echo "[run_parallel] GPU GATE WARN: probe failed (rc=${probe_rc}: ${gpu_free:-N/A}); passing through." >&2
                break
            fi
            if (( gpu_free >= gpu_gate_mib )); then
                break
            fi
            echo "[run_parallel] GPU GATE: ${gpu_free} MiB free < ${gpu_gate_mib} MiB threshold; sleeping 30s and re-probing (attempt $((probe_attempts + 1))/60)" >&2
            sleep 30
            probe_attempts=$((probe_attempts + 1))
        done
        if [[ ${probe_attempts} -ge 60 ]]; then
            echo "[run_parallel] GPU GATE WARN: gave up after 60 retries (~30 min); proceeding anyway." >&2
        fi
    fi
    # Concurrency gate: block if BG_PIDS already has STONKBENCH_LOCAL_JOBS
    # alive children. While blocked, `wait -n` reaps one child and we
    # (a) record any non-zero rc into REAPED_FAILURES so wait_for_jobs
    #     surfaces it at stage end (otherwise it would be silently dropped
    #     when the kill -0 filter below removes the reaped PID from
    #     BG_PIDS), and (b) drop already-reaped PIDs from BG_PIDS so the
    #     loop guard stays accurate.
    if [[ "${STONKBENCH_LOCAL_JOBS}" != "0" ]]; then
        while [[ ${#BG_PIDS[@]} -ge ${STONKBENCH_LOCAL_JOBS} ]]; do
            local rc=0
            wait -n || rc=$?
            if [[ "${rc}" -ne 0 && "${rc}" -ne 127 ]]; then
                # ISO-8601 UTC clock at the moment of reap so a log-tail
                # operator can correlate this WARN with the corresponding
                # trial log without grepping timestamps separately.
                REAPED_FAILURES+=("$(date -u +%T) gate_rc=${rc}")
            fi
            local alive=() j_pid
            if [[ ${#BG_PIDS[@]} -gt 0 ]]; then
                for j_pid in "${BG_PIDS[@]}"; do
                    if kill -0 "${j_pid}" 2>/dev/null; then alive+=("${j_pid}"); fi
                done
            fi
            BG_PIDS=("${alive[@]}")
        done
    fi
    bash "${RUN_ON_GPU}" "${slot}" "$@" > "${log}" 2>&1 &
    BG_PIDS+=($!)
}

# wait_for_jobs <label>: drain BG_PIDS, propagating failures. Returns 0 on
# all-pass, non-zero if any child exited non-zero. Also drains the
# REAPED_FAILURES accumulator populated by launch()'s gate — a child that
# died while we were waiting to enqueue the next one is captured there.
wait_for_jobs() {
    local label="$1"
    # Emit the "+ M gate-reaped" suffix only when non-zero so the happy
    # path doesn't pollute every banner with a "+ 0 gate-reaped" token.
    local gate_suffix=""
    if [[ ${#REAPED_FAILURES[@]} -gt 0 ]]; then
        gate_suffix=" + ${#REAPED_FAILURES[@]} gate-reaped"
    fi
    echo "[run_parallel] waiting for ${label} jobs (${#BG_PIDS[@]} pending${gate_suffix})..."
    # Surface in-gate failures first; clear so the next stage starts fresh.
    if [[ ${#REAPED_FAILURES[@]} -gt 0 ]]; then
        echo "[run_parallel] ERROR: ${label} gate reaped ${#REAPED_FAILURES[@]} failed child(ren) before drain: ${REAPED_FAILURES[*]}" >&2
        echo "[run_parallel] inspect logs under ${OUTPUT_ROOT}/logs/${RUN_ID}/${label}/" >&2
        REAPED_FAILURES=()
        return 1
    fi
    local failed_pids=()
    local skipped_pids=()
    if [[ ${#BG_PIDS[@]} -gt 0 ]]; then
        local pid rc
        for pid in "${BG_PIDS[@]}"; do
            if ! wait "${pid}"; then
                rc=$?
                # rc=127 means the PID is no longer our child (already reaped
                # by a sibling or parent signal). Treat as non-fatal but track
                # and report the count so debugging isn't silent. Real
                # failures (rc != 0 && rc != 127) accumulate into failed_pids.
                if [[ "${rc}" -eq 127 ]]; then
                    skipped_pids+=("${pid}")
                else
                    failed_pids+=("${pid}:rc=${rc}")
                fi
            fi
        done
    fi
    BG_PIDS=()
    REAPED_FAILURES=()
    if [[ "${#skipped_pids[@]}" -gt 0 ]]; then
        echo "[run_parallel] WARN: ${label} rc=127 (already reaped) for ${#skipped_pids[@]} pid(s) — skipped" >&2
    fi
    if [[ "${#failed_pids[@]}" -gt 0 ]]; then
        echo "[run_parallel] ERROR: ${label} failed for ${#failed_pids[@]} child process(es)" >&2
        echo "[run_parallel] PIDs: ${failed_pids[*]}" >&2
        echo "[run_parallel] inspect logs under ${OUTPUT_ROOT}/logs/${RUN_ID}/${label}/" >&2
        return 1
    fi
    echo "[run_parallel] ${label} complete"
}

stage_hp_search() {
    local out_dir="${OUTPUT_ROOT}/results/${RUN_ID}/hp_search"
    mkdir -p "${out_dir}"
    local flag=()
    [[ "${SMOKE}" == "1" ]] && flag+=(--smoke)
    local i=0
    for tid in $(seq 0 $((N_HP_TRIALS - 1))); do
        local slot=$(( i % N_GPUS ))
        launch "${slot}" "${OUTPUT_ROOT}/logs/${RUN_ID}/hp_search/trial_${tid}.log" \
            python -m src.experiments.hp_search \
                --device cuda --trial_id "${tid}" \
                --output_dir "${out_dir}" "${flag[@]}"
        i=$(( i + 1 ))
    done
    if ! wait_for_jobs "HP search (${N_HP_TRIALS} trials)"; then
        return 1
    fi
    echo "[run_parallel] aggregating HP trials..."
    python -m src.experiments.hp_search --aggregate_only --output_dir "${out_dir}"
}

stage_aggregate() {
    local out_dir="${OUTPUT_ROOT}/results/${RUN_ID}/hp_search"
    echo "[run_parallel] aggregating HP trials under ${out_dir}"
    python -m src.experiments.hp_search --aggregate_only --output_dir "${out_dir}"
}

stage_final_train() {
    local hp_summary="${OUTPUT_ROOT}/results/${RUN_ID}/hp_search/summary.json"
    if [[ ! -f "${hp_summary}" ]]; then
        echo "[run_parallel] WARN: no HP summary at ${hp_summary}; writing empty one for stats models."
        mkdir -p "$(dirname "${hp_summary}")"
        echo '{"models": {}}' > "${hp_summary}"
    fi
    local smoke_args=()
    [[ "${SMOKE}" == "1" ]] && smoke_args+=(--smoke)
    local i=0
    for m in "${MODELS_ALL[@]}"; do
        local slot=$(( i % N_GPUS ))
        launch "${slot}" "${OUTPUT_ROOT}/logs/${RUN_ID}/final_train/${m}.log" \
            python -m src.experiments.run_final_training \
                --models "${m}" \
                --hp_summary "${hp_summary}" \
                --output_root "${OUTPUT_ROOT}" \
                --run_id "${RUN_ID}" \
                --generation_length "${GENERATION_LENGTH}" \
                --num_samples "${EFFECTIVE_NUM_SAMPLES}" \
                --seed "${SEED}" \
                --device cuda \
                "${smoke_args[@]}"
        i=$(( i + 1 ))
    done
    if ! wait_for_jobs "Final train (14 models)"; then
        return 1
    fi
}

# Eval runs SERIALLY. unified_evaluator.UnifiedEvaluator.run() iterates over
# every artifact under --generated_dir and writes per-artifact metrics.json
# AND a shared complete_evaluation.json summary. Multiple parallel jobs on
# the same --results_dir would corrupt the summary. So we run a single
# unified_evaluator instance on the whole artifacts tree.
stage_eval() {
    local artifacts_root="${OUTPUT_ROOT}/results/${RUN_ID}"
    if [[ ! -d "${artifacts_root}" ]]; then
        echo "[run_parallel] no artifacts under ${artifacts_root}; run final_train first." >&2
        exit 1
    fi
    local eval_dir="${OUTPUT_ROOT}/evaluation/${RUN_ID}"
    mkdir -p "${eval_dir}"

    local seqs="${SEQ_LENGTHS[*]}"
    echo "[run_parallel] evaluating all artifacts under ${artifacts_root} (serial)..."
    # Single sequential call. We still wrap in run_on_gpu.sh so the eval
    # inherits CUDA_VISIBLE_DEVICES pin (slot 0 by default) and the same
    # thread clamping as parallel jobs. Without the wrapper, the legacy
    # vendor UtilityEvaluator would pick cuda:0 itself and could oversubscribe
    # the host thread pool inherited from the user's shell.
    if ! bash "${RUN_ON_GPU}" 0 python -m src.unified_evaluator \
        --generated_dir "${artifacts_root}" \
        --results_dir "${eval_dir}" \
        --seq_lengths ${seqs} \
        --skip_regenerate \
        > "${OUTPUT_ROOT}/logs/${RUN_ID}/eval/unified_evaluator.log" 2>&1; then
        echo "[run_parallel] ERROR: unified_evaluator failed; see ${OUTPUT_ROOT}/logs/${RUN_ID}/eval/unified_evaluator.log" >&2
        return 1
    fi

    # Aggregate eval summary (no GPUS, CPU only).
    python - <<PY
import json, pathlib
root = pathlib.Path("${eval_dir}").resolve()
counts = {}
for p in root.rglob("metrics.json"):
    parts = p.relative_to(root).parts
    if not parts:
        continue
    counts[parts[0]] = counts.get(parts[0], 0) + 1
print("[run_parallel] eval summary:")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v} artifact(s)")
PY
}

case "${STAGE}" in
    hp_search)  stage_hp_search ;;
    aggregate)  stage_aggregate ;;
    final_train) stage_final_train ;;
    eval)       stage_eval ;;
    all)
        if ! stage_hp_search; then
            echo "[run_parallel] FATAL: hp_search stage failed; aborting all." >&2
            exit 1
        fi
        if ! stage_final_train; then
            echo "[run_parallel] FATAL: final_train stage failed; aborting all." >&2
            exit 1
        fi
        if ! stage_eval; then
            echo "[run_parallel] FATAL: eval stage failed; aborting all." >&2
            exit 1
        fi
        ;;
    *) usage >&2; exit 1 ;;
esac

echo "[run_parallel] stage '${STAGE}' finished. OUTPUT_ROOT=${OUTPUT_ROOT} RUN_ID=${RUN_ID}"
