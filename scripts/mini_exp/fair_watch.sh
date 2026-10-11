#!/bin/bash
# Watcher for fair_20261004: waits for the two fair-train jobs to leave the
# queue, then runs the FINAL collapse/realism scoring over the complete
# results set + rebuilds report.md/curves. Log: /tmp/fair_watch.log
set -u
LOG=/tmp/fair_watch.log
RUN_ID=fair_20261004
REPORT_DIR=/home/epham/StonkBench/outputs/mini_exp/${RUN_ID}
OUTPUT_ROOT=/home/epham/StonkBench/outputs
PY="$HOME/.venvs/stonkbench/bin/python"
JOBS="1031776,1031588"

: > "$LOG"
echo "$(date -Is) WATCH_START jobs=$JOBS" >> "$LOG"
done_flag=0
for i in $(seq 1 240); do
  st=$(ssh -o BatchMode=yes trig-login01 "squeue -h -j ${JOBS} -o '%T'" 2>/dev/null | tr '\n' ',')
  echo "$(date -Is) queue=[${st}]" >> "$LOG"
  if [ -z "${st}" ]; then done_flag=1; break; fi
  sleep 120
done
if [ "${done_flag}" != 1 ]; then
  echo "$(date -Is) WATCH_TIMEOUT" >> "$LOG"
  exit 1
fi
echo "$(date -Is) JOBS_DONE - scoring complete results" >> "$LOG"
cd "$HOME/StonkBench" || exit 1
set +e
"$PY" scripts/mini_exp/score_outputs.py \
  --run_id "${RUN_ID}" \
  --output_root "${OUTPUT_ROOT}" \
  --report_dir "${REPORT_DIR}" >> "$LOG" 2>&1
echo "$(date -Is) SCORE_EXIT=$?" >> "$LOG"
"$PY" scripts/mini_exp/refresh_report.py --run_dir "${REPORT_DIR}" >> "$LOG" 2>&1
echo "$(date -Is) REFRESH_EXIT=$?" >> "$LOG"
echo "$(date -Is) WATCH_COMPLETE" >> "$LOG"
