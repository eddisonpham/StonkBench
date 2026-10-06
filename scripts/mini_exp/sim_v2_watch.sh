#!/bin/bash# Watcher for array jobs 1033384 + 1033486 + 1033541 (simulation_v2 native
# retraining; retries after transient CANCELLED-0:54 watchdog kills and the
# timegrad native-fit fix). On completion: task-rc summary, artifact count,
# NATIVE-length
 metadata
# check (train_sequence_length == sequence_length == L => no stitching),
# results/latest + fair_20261004 integrity, scoring, simulation_v2 plots.
# Log: /tmp/simv2_watch.log
set -u
LOG=/tmp/simv2_watch.log
RUN_ID=simulation_v2
PY="$HOME/.venvs/stonkbench/bin/python"
REPO="$HOME/StonkBench"
REPORT_DIR=/scratch/epham/stonkbench/mini_exp/${RUN_ID}
OUTPUT_ROOT=/scratch/epham/stonkbench/output

: > "$LOG"
echo "$(date -Is) WATCH_START jobs=1033384,1033486,1033541" >> "$LOG"
JOBS="1033384,1033486,1033541"
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
echo "$(date -Is) ARRAYS_LEFT_QUEUE" >> "$LOG"

# task rc summary from the per-task logs
ssh -o BatchMode=yes trig-login01 \
  'grep -h "=== SIMV2 DONE" /scratch/epham/stonkbench/slurm_logs/simv2_*.out 2>/dev/null' \
  >> "$LOG" 2>&1
ok=$(grep -c "rc=0 " "$LOG" || true)
bad=$(grep -c "SIMV2 DONE" "$LOG")
echo "$(date -Is) TASKS_DONE=${bad} RC0=${ok}" >> "$LOG"

cd "$REPO" || exit 1
set +e

# 1. artifacts + NATIVE-length check (train_sequence_length == L == seq)
"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import glob, torch
rows, bad_native, missing = [], [], []
expect = {}
for m in ["unconditional_tsdiffusion","conditional_tsdiffusion","quantgan",
          "timegrad","vrnn_klsched","kalman_vae_safe","cond_sig_wgan"]:
    for L in (21,42,126,252):
        expect[f"{m}_seq{L}.pt"] = L
found = set()
for p in sorted(glob.glob('/scratch/epham/stonkbench/output/results/simulation_v2/*/artifacts/*.pt')):
    name = p.split('/')[-1]
    found.add(name)
    raw = torch.load(p, map_location='cpu', weights_only=False)
    md = raw.get('metadata', {}) if isinstance(raw, dict) else {}
    data = raw['data'] if isinstance(raw, dict) else raw
    L = expect.get(name)
    sl, tl = md.get('sequence_length'), md.get('train_sequence_length')
    native = (L is not None and sl == L and tl == L and data.shape[1] == L)
    rows.append((name, sl, tl, tuple(data.shape), md.get('num_samples'), native))
    if not native:
        bad_native.append(name)
missing = sorted(set(expect) - found)
for r in rows:
    print(f"ART {r[0]:44s} seq_len={r[1]} train_len={r[2]} shape={r[3]} n={r[4]} native={'Y' if r[5] else 'N'}")
print(f"ARTIFACTS={len(rows)}/28 MISSING={len(missing)} NON_NATIVE={len(bad_native)}")
if missing:
    print("MISSING:", missing)
if bad_native:
    print("NON_NATIVE:", bad_native)
print("NATIVE_ALL_OK" if not bad_native and not missing else "NATIVE_CHECK_FAILED")
PYEOF
echo "$(date -Is) NATIVE_RC=$?" >> "$LOG"

# 2. integrity of pre-existing trees
find "${OUTPUT_ROOT}/results/latest" "${OUTPUT_ROOT}/results/fair_20261004" \
     -type f -printf "%p %s %T@\n" 2>/dev/null | sort > /tmp/v2_after.txt
if diff -q /tmp/v2_baseline.txt /tmp/v2_after.txt >> "$LOG" 2>&1; then
  echo "$(date -Is) PREEXISTING_INTEGRITY=UNCHANGED" >> "$LOG"
else
  echo "$(date -Is) PREEXISTING_INTEGRITY=CHANGED" >> "$LOG"
  diff /tmp/v2_baseline.txt /tmp/v2_after.txt | head -20 >> "$LOG"
fi

# 3. score the simulation_v2 run
"$PY" scripts/mini_exp/score_outputs.py \
  --run_id "${RUN_ID}" \
  --output_root "${OUTPUT_ROOT}" \
  --report_dir "${REPORT_DIR}" >> "$LOG" 2>&1
echo "$(date -Is) SCORE_EXIT=$?" >> "$LOG"

# 4. simulation_v2 overlay plots
"$PY" scripts/mini_exp/plot_simulations.py \
  --run_id "${RUN_ID}" \
  --output_root "${OUTPUT_ROOT}" \
  --report_dir "${REPORT_DIR}" >> "$LOG" 2>&1
echo "$(date -Is) PLOT_EXIT=$?" >> "$LOG"
echo "$(date -Is) WATCH_COMPLETE" >> "$LOG"
