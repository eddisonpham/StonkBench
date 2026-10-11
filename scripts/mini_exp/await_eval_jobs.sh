#!/bin/bash
# Report-only progress monitor for scripts/mini_exp/eval_gen_batches.sh.
#
#   bash scripts/mini_exp/await_eval_jobs.sh <array_job_id[,id...]> [logfile] [root]
#
# Multiple job ids may be watched at once (comma separated) — e.g. a long-lived
# array plus its short-walltime re-submission.
#
# Prints one line per poll: how many of the 432 cells have a complete per-cell
# summary and how many array tasks are running/pending. Exits 0 once all cells
# are present.
#
# Failed/timed-out task states are reported as a warning when the set changes,
# but do not stop the monitor: sacct history is sticky, so a task that already
# timed out and was re-queued elsewhere (e.g. 1040158_26 -> job 1040509) would
# otherwise kill this watcher on its first poll. The authoritative gate on
# completeness is check_eval_coverage.py / finalize_eval.sh, which refuses to
# aggregate a grid with missing cells.
#
# It does NOT resubmit anything. An earlier version of this script resubmitted
# "abandoned" tasks on every poll and piled up a dozen duplicate array jobs, so
# resubmission is deliberately left to a human.
#
# A cell counts as done only when
#   <root>/<ablation>/evaluation_batch<k>/<model>_seq<L>/complete_evaluation.json
# exists and holds exactly one non-empty entry keyed by that directory name.
set -uo pipefail

JOB="${1:?usage: await_eval_jobs.sh <array_job_id> [logfile] [root]}"
LOG="${2:-/tmp/await_eval_${JOB}.log}"
ROOT="${3:-/home/epham/StonkBench/outputs/generation}"
INTERVAL="${INTERVAL:-300}"
EXPECTED="${EXPECTED:-432}"

# Scheduler queries must go through the submission host: this may be a
# different login node (tri-login05) whose slurm controller does not own the
# job. Output lives on a filesystem shared by both login nodes.
SQ() { ssh -o BatchMode=yes trig-login01 "$*"; }

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

log "watching job ${JOB}; expecting ${EXPECTED} cells under ${ROOT}"
while true; do
  DONE="$(
    python - "${ROOT}" <<'PY'
import json, sys
from pathlib import Path

count = 0
for path in Path(sys.argv[1]).glob("*/evaluation_batch*/*_seq*/complete_evaluation.json"):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    # exactly the one cell this directory is named after, and non-empty
    if list(data) == [path.parent.name] and next(iter(data.values())):
        count += 1
print(count)
PY
  )"
  STATES="$(SQ "squeue -j ${JOB} -h -o '%t'" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
  log "done=${DONE}/${EXPECTED} queue[${STATES:-empty}]"

  if [[ "${DONE}" -ge "${EXPECTED}" ]]; then
    log "all ${EXPECTED} cells complete"
    exit 0
  fi

  # CANCELLED is deliberately NOT a failure: tasks are cancelled on purpose
  # when the layout/walltime changes and are re-submitted elsewhere. FAILED /
  # TIMEOUT states in sacct are historical and sticky, so warn only when the
  # set changes instead of exiting (see header). The coverage check in
  # finalize_eval.sh is what refuses an incomplete grid.
  BAD="$(SQ "sacct -j ${JOB} -X -n -o State" 2>/dev/null | sort -u | grep -E 'FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY' | tr '\n' ' ')"
  if [[ "${BAD}" != "${BAD_SEEN:-}" ]]; then
    log "task state(s) ${BAD:-<none>}present — re-queued cells are covered; coverage check remains the gate"
    BAD_SEEN="${BAD}"
  fi

  sleep "${INTERVAL}"
done
