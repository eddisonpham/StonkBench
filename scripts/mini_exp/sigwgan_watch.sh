#!/bin/bash
# Watcher for job 1032854 (cond_sig_wgan regeneration with unique-sample fix).
# On completion: uniqueness check, results/latest integrity diff vs snapshot,
# final scoring. Log: /tmp/sigwgan_watch.log
set -u
LOG=/tmp/sigwgan_watch.log
RUN_ID=fair_20261004
PY="$HOME/.venvs/stonkbench/bin/python"
REPORT_DIR=/scratch/epham/stonkbench/mini_exp/${RUN_ID}
OUTPUT_ROOT=/scratch/epham/stonkbench/output

: > "$LOG"
echo "$(date -Is) WATCH_START job=1032854" >> "$LOG"
done_flag=0
for i in $(seq 1 90); do
  st=$(ssh -o BatchMode=yes trig-login01 "squeue -h -j 1032854 -o '%T'" 2>/dev/null | tr '\n' ',')
  echo "$(date -Is) queue=[${st}]" >> "$LOG"
  if [ -z "${st}" ]; then done_flag=1; break; fi
  sleep 30
done
if [ "${done_flag}" != 1 ]; then
  echo "$(date -Is) WATCH_TIMEOUT" >> "$LOG"
  exit 1
fi
echo "$(date -Is) JOB_LEFT_QUEUE" >> "$LOG"
ssh -o BatchMode=yes trig-login01 'grep -E "TRAIN DONE|SCORE rc|REFRESH_REPORT rc|Traceback" /scratch/epham/stonkbench/slurm_logs/fair_train_1032854.out | tail -6' >> "$LOG" 2>&1

# 1. uniqueness of regenerated artifacts
cd "$HOME/StonkBench" || exit 1
set +e
"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import numpy as np, torch, glob
ok = True
for p in sorted(glob.glob('/scratch/epham/stonkbench/output/results/fair_20261004/cond_sig_wgan/artifacts/*.pt')):
    raw = torch.load(p, map_location='cpu', weights_only=False)
    arr = raw['data'].float().numpy() if isinstance(raw, dict) else np.asarray(raw)
    flat = arr.reshape(arr.shape[0], -1)
    uniq = np.unique(np.round(flat, 10), axis=0).shape[0]
    status = "OK" if uniq == flat.shape[0] else "DUP!"
    if uniq != flat.shape[0]:
        ok = False
    print(f"UNIQUE {status} {p.split('/')[-1]}: {uniq}/{flat.shape[0]}")
print("UNIQUE_ALL_OK" if ok else "UNIQUE_FAILED")
PYEOF
echo "$(date -Is) UNIQUE_RC=$?" >> "$LOG"

# 2. results/latest integrity vs pre-job snapshot
find "${OUTPUT_ROOT}/results/latest" "${OUTPUT_ROOT}/logs/timegrad" \
     "${OUTPUT_ROOT}/sanity/latest" -type f -printf "%p %s %T@\n" 2>/dev/null \
  | sort > /tmp/latest_snapshot_after.txt
if diff -q /tmp/latest_snapshot_before.txt /tmp/latest_snapshot_after.txt >> "$LOG" 2>&1; then
  echo "$(date -Is) LATEST_INTEGRITY=UNCHANGED" >> "$LOG"
else
  echo "$(date -Is) LATEST_INTEGRITY=CHANGED" >> "$LOG"
  diff /tmp/latest_snapshot_before.txt /tmp/latest_snapshot_after.txt | head -20 >> "$LOG"
fi

# 3. final scoring over the complete run
"$PY" scripts/mini_exp/score_outputs.py \
  --run_id "${RUN_ID}" \
  --output_root "${OUTPUT_ROOT}" \
  --report_dir "${REPORT_DIR}" >> "$LOG" 2>&1
echo "$(date -Is) SCORE_EXIT=$?" >> "$LOG"
"$PY" scripts/mini_exp/refresh_report.py --run_dir "${REPORT_DIR}" >> "$LOG" 2>&1
echo "$(date -Is) REFRESH_EXIT=$?" >> "$LOG"
echo "$(date -Is) WATCH_COMPLETE" >> "$LOG"
