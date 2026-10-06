#!/bin/bash
# Live monitor for a mini-experiment run: queue state + live training curves
# + aggregated report. Run on the login node while jobs execute.
#
#   bash scripts/mini_exp/monitor.sh <run_id> [--follow]
set -euo pipefail
RUN_ID="${1:?usage: monitor.sh <run_id> [--follow]}"
FOLLOW="${2:-}"
MINI_ROOT="${STONKBENCH_MINI_ROOT:-/scratch/epham/stonkbench/mini_exp}"
RUN_DIR="${MINI_ROOT}/${RUN_ID}"

echo "=== queue ==="
squeue -u "${USER}" -o "%.10i %.12j %.8T %.10M %.10l" | grep -E "mini-dl|JOBID" || true

echo
echo "=== events so far (per-model epoch counts) ==="
if [[ -d "${RUN_DIR}/events" ]]; then
  for f in "${RUN_DIR}"/events/*.jsonl; do
    [[ -e "$f" ]] || continue
    printf "%-28s %s epochs\n" "$(basename "$f")" "$(grep -c '"event": "epoch"' "$f" || true)"
  done
else
  echo "(no events dir yet: ${RUN_DIR}/events)"
fi

echo
python "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/mini_exp/refresh_report.py" \
  --run_dir "${RUN_DIR}" 2>/dev/null || echo "(report not ready yet)"

if [[ "${FOLLOW}" == "--follow" ]]; then
  echo
  echo "=== following live curves (ctrl-c to stop) ==="
  tail -n +1 -f "${RUN_DIR}"/events/*.jsonl 2>/dev/null || true
fi
