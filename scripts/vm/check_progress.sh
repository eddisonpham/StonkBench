#!/usr/bin/env bash
# scripts/vm/check_progress.sh
#
# One-shot status reporter for the running StonkBench pipeline. Designed to be
# called REPEATEDLY from a separate terminal while `scripts/vm/run_parallel.sh
# all` is running.
#
# Reports:
#   (1) tmux session state (alive / dead / not-starting)
#   (2) per-phase job counts (HP trials; final_train models; eval artifacts)
#   (3) per-model wall-clock seconds pulled from `events.jsonl`
#   (4) GPU utilization from `nvidia-smi`
#   (5) disk usage of the output root
#
# Usage:
#   bash scripts/vm/check_progress.sh [STONKBENCH_RUN_ID]
#     # default: read STONKBENCH_RUN_ID env, else the most recent dated dir
#   watch -n 60 bash scripts/vm/check_progress.sh
#     # for a 60s refresh in the same terminal

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/phamnhut/StonkBench}"
OUTPUT_ROOT="${STONKBENCH_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"

# Resolve RUN_ID (CLI arg > env > latest dated subdir of $OUTPUT_ROOT/results).
if [[ $# -ge 1 ]]; then
  RUN_ID="$1"
else
  RUN_ID="${STONKBENCH_RUN_ID:-}"
fi
if [[ -z "${RUN_ID}" && -d "${OUTPUT_ROOT}/results" ]]; then
  RUN_ID="$(ls -1 "${OUTPUT_ROOT}/results" 2>/dev/null | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -r | head -1 || true)"
fi

LOG_DIR="${OUTPUT_ROOT}/logs/${RUN_ID}"
HP_DIR="${OUTPUT_ROOT}/results/${RUN_ID}/hp_search"
ART_DIR="${OUTPUT_ROOT}/results/${RUN_ID}"
CKPT_DIR="${OUTPUT_ROOT}/checkpoints/${RUN_ID}"
EVAL_DIR="${OUTPUT_ROOT}/evaluation/${RUN_ID}/seq_252"
SANITY_DIR="${OUTPUT_ROOT}/sanity/${RUN_ID}"

# Pretty-divider helper.
section() { echo; echo "==[ $* ]=========================================================="; }

section "StonkBench pipeline status"
echo "  PROJECT_ROOT      = ${PROJECT_ROOT}"
echo "  OUTPUT_ROOT       = ${OUTPUT_ROOT}"
echo "  STONKBENCH_RUN_ID = ${RUN_ID:-<not resolved - no results dir>}"

section "tmux session state"
if command -v tmux >/dev/null 2>&1; then
  tmux list-sessions 2>/dev/null | grep -E '^stonkbench_run' || echo "  (no stonkbench_run tmux session)"
else
  echo "  (tmux not installed)"
fi

section "HP search progress"
if [[ -d "${HP_DIR}/trials" ]]; then
  HP_DONE=$(ls -1 "${HP_DIR}/trials"/*.json 2>/dev/null | wc -l)
  # N_HP_TRIALS is computed by run_parallel.sh; we re-compute it here for the reader.
  N_HP_TOTAL=$(/home/phamnhut/miniconda3/envs/stonk/bin/python - <<'PY' 2>/dev/null || echo 72
from src.experiments.hp_configs import DL_MODEL_KEYS, configs_for_model
print(sum(len(configs_for_model(m)) for m in DL_MODEL_KEYS))
PY
)
  # Read summary.json if it exists (post-aggregate).
  if [[ -f "${HP_DIR}/summary.json" ]]; then
    HP_DONE=$(/home/phamnhut/miniconda3/envs/stonk/bin/python -c "import json; d=json.load(open('${HP_DIR}/summary.json')); print(sum(len(d['models'].get(m, {}).get('ranked_configs', [])) for m in d.get('models', {})))" 2>/dev/null || echo "${HP_DONE}")
  fi
  printf "  trials completed : %s / %s\n" "${HP_DONE}" "${N_HP_TOTAL}"
  LAST_TRIAL=$(ls -1t "${HP_DIR}/trials"/*.json 2>/dev/null | head -1 | xargs -I{} basename {} 2>/dev/null || echo none)
  echo "  most recent trial: ${LAST_TRIAL}"
else
  echo "  (hp_search not yet started OR no RUN_ID)"
fi
if [[ -d "${LOG_DIR}/hp_search" ]]; then
  HP_RUNNING=$(ls -1 "${LOG_DIR}/hp_search"/trial_*.log 2>/dev/null | wc -l)
  HP_DONE_LOGS=$(grep -lE "EVENT:RUN_END" "${LOG_DIR}/hp_search"/trial_*.log 2>/dev/null | wc -l || echo 0)
  printf "  trial logs       : %s running, %s completed\n" "${HP_RUNNING}" "${HP_DONE_LOGS}"
fi

section "Final training progress"
if [[ -d "${ART_DIR}" ]]; then
  # Each model final-train drops a {model_key}_seq_252.pt inside results/<rid>/<model>/artifacts.
  ARTIFACTS=$(find "${ART_DIR}" -mindepth 3 -maxdepth 3 -name "*_seq_252.pt" -printf '%f\n' 2>/dev/null | sort -u)
  if [[ -n "${ARTIFACTS}" ]]; then
    echo "  artifacts produced:"
    while IFS= read -r a; do echo "    - $a"; done <<< "${ARTIFACTS}"
  else
    echo "  (no artifacts yet)"
  fi
elif [[ -d "${LOG_DIR}" ]]; then
  # Reconstruct from log dir.
  for log in "${LOG_DIR}"/final_train/*.log; do
    [[ -f "${log}" ]] || continue
    model="$(basename "${log}" .log)"
    echo "  ${model}: $(grep -c 'EVENT:RUN_END' "${log}" 2>/dev/null || echo 0) run_end events in log"
  done
else
  echo "  (final_train not yet started)"
fi

section "Per-model wall-clock (from events.jsonl)"
if [[ -d "${LOG_DIR}" ]]; then
  for model_dir in "${LOG_DIR}"/*/; do
    [[ -f "${model_dir}/events.jsonl" ]] || continue
    model="$(basename "${model_dir}")"
    last_ended="$(grep '"event":"run_end"' "${model_dir}/events.jsonl" 2>/dev/null | tail -1)"
    if [[ -n "${last_ended}" ]]; then
      /home/phamnhut/miniconda3/envs/stonk/bin/python -c "
import json,sys
line='${last_ended}'
try:
    rec=json.loads(line)
    print(f\"  {rec.get('model_key','${model}'):24s} elapsed_sec={rec.get('elapsed_sec',0):8.1f}\")
except Exception as e:
    print(f\"  ${model}: parse-error {e}\")
" 2>/dev/null || echo "  ${model}: (unparseable)"
    else
      echo "  ${model}: still RUNNING or not started"
    fi
  done
fi

section "Evaluation progress"
if [[ -d "${EVAL_DIR}" ]]; then
  N_METRICS=$(find "${EVAL_DIR}" -name 'metrics.json' 2>/dev/null | wc -l)
  echo "  metrics.json files: ${N_METRICS}"
else
  echo "  (eval not yet started)"
fi
if [[ -d "${SANITY_DIR}" ]]; then
  N_SANITY=$(find "${SANITY_DIR}" -name 'per_channel_summary.csv' 2>/dev/null | wc -l)
  echo "  per-channel summaries: ${N_SANITY} model(s) already drew all-channel sanity viz"
fi

section "GPU utilization"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo '  (nvidia-smi query failed)'
else
  echo '  (nvidia-smi not installed)'
fi

section "Disk usage"
if [[ -d "${OUTPUT_ROOT}" ]]; then
  du -sh "${OUTPUT_ROOT}"/* 2>/dev/null | sort -h | tail -10
fi

section "Latest log lines"
LATEST_LOG="$(ls -1t "${LOG_DIR}"/final_train/*.log "${LOG_DIR}"/hp_search/*.log "${LOG_DIR}"/eval/*.log 2>/dev/null | head -1 || true)"
if [[ -n "${LATEST_LOG}" ]]; then
  echo "  (tail ${LATEST_LOG})"
  tail -n 5 "${LATEST_LOG}"
else
  echo "  (no per-job log files yet)"
fi
echo
