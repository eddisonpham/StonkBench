#!/bin/bash
#SBATCH --job-name=geneval
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-431
#SBATCH --output=/home/%u/StonkBench/slurm_logs/geneval_%A_%a.out
#SBATCH --error=/home/%u/StonkBench/slurm_logs/geneval_%A_%a.err

# ============================================================================
# End-to-end evaluation of the multi-batch generation trees.
#
# ONE ARTIFACT = ONE JOB. Array 0-431 covers the full grid
#   3 ablations x 3 batches x 12 models x 4 sequence lengths = 432 cells
#
# Index map (tid -> cell):
#   seq_idx   =  tid      % 4     -> 21 / 42 / 126 / 252
#   model_idx = (tid / 4) % 12    -> MODELS[]
#   batch     = (tid / 48) % 3 +1 -> batch1 .. batch3
#   ablation  =  tid / 144        -> simulation_v2 / fair_20261004 / latest
#
# Each task evaluates exactly one artifact end to end:
#   fidelity / diversity / stylized facts / visual + the downstream metrics
#   the evaluator owns (utility + hedging, portfolio optimisation, P&L),
#   then generates that cell's plots.
#
# Every task writes into its OWN results directory
#   <GEN_ROOT>/<ablation>/evaluation_batch<k>/<model>_seq<L>/
# because unified_evaluator.run() OVERWRITES complete_evaluation.json on every
# invocation. One cell per directory means every complete_evaluation.json holds
# exactly one entry, writers never race, and there is no shard-merge step:
# the aggregation reads the per-cell summaries directly
# (scripts/mini_exp/aggregate_batch_metrics.py).
#
# Walltime: a single n=500 DL cell measures ~18 min (job 1034399: 28 artifacts
# = 9:29 h), statistical cells a few minutes; 6 h leaves a wide margin while
# staying inside the partition's 24 h ceiling and keeping tasks short enough to
# backfill on a contended cluster.
#
# NO --seq_lengths (plural) filter: that mode trims the LAST L steps off the
# artifact. --seq_length (singular, passed below) keeps only the artifact whose
# NATIVE length equals L, which is what we want — every artifact is already
# stored at its target length.
#
# Real-data side: common.sh pins STONKBENCH_DL_SET_PATH to data/preprocessed/
# dl_set.pt (the W252 symlink) — identical to the simulation_v2 eval protocol
# that produced the accepted PASS=19 / WARN=9 scorecard, so results stay
# comparable.
#
# Single-cell overrides (each independent, e.g. for smoke tests):
#   EVAL_ABLATION=fair_20261004 EVAL_BATCH=2 EVAL_SEQ=126 EVAL_MODEL=timegrad
#   EVAL_RESULTS_DIR=<path>   # redirect the results_dir (smoke runs)
#   GEN_ROOT=<path>           # default ${OUTPUT_ROOT}/generation
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

# Thread clamps — REQUIRED, do not remove. The generation drivers already set
# these (gen_batches.sh: 8, gen_batches_stat.sh: 4) but this script originally
# did not, and without them torch/OpenMP size their pools from the NODE's core
# count (~96) instead of the 8 CPUs slurm allocated. The oversized pool
# spin-waits inside the 8-CPU affinity mask, starving the working thread: the
# affected tasks burned 7-9 CPU-hours per artifact and crawled (a seq126 shard
# needed 80 min for an artifact that takes 5 min unthrottled, which would have
# blown the walltime). Keep the pool equal to cpus-per-task and make the wait
# policy passive so idle threads sleep instead of spinning.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-passive}"

ABLATIONS=(simulation_v2 fair_20261004 latest)
MODELS=(block_bootstrap conditional_tsdiffusion cond_sig_wgan de_jump_diffusion
        dcc_garch kalman_vae_safe merton_jump_diffusion quantgan
        stationary_block_bootstrap timegrad unconditional_tsdiffusion
        vrnn_klsched)
SEQS=(21 42 126 252)

idx_of() {  # idx_of <needle> <haystack...>
  local needle="$1" i=0 hay
  shift
  for hay in "$@"; do
    if [[ "${hay}" == "${needle}" ]]; then echo "${i}"; return 0; fi
    i=$(( i + 1 ))
  done
  echo "[FATAL] unknown value '${needle}' (valid: $*) >&2" >&2
  return 1
}

TID="${SLURM_ARRAY_TASK_ID:-0}"
REM="${TID}"

SEQ_I=$(( REM % 4 ));  REM=$(( REM / 4 ))
MODEL_I=$(( REM % 12 )); REM=$(( REM / 12 ))
BATCH_I=$(( REM % 3 ));  REM=$(( REM / 3 ))
ABL_I="${REM}"

SEQ="${EVAL_SEQ:-${SEQS[${SEQ_I}]}}"
MODEL="${EVAL_MODEL:-${MODELS[${MODEL_I}]}}"
BATCH="${EVAL_BATCH:-$(( BATCH_I + 1 ))}"
ABLATION="${EVAL_ABLATION:-${ABLATIONS[${ABL_I}]}}"

# Re-derive indices when overridden, so the log line is unambiguous and the
# derived results_dir cannot silently point at the wrong cell.
[[ -n "${EVAL_SEQ:-}" ]]  && SEQ_I="$( idx_of "${SEQ}" "${SEQS[@]}" )"
[[ -n "${EVAL_MODEL:-}" ]] && MODEL_I="$( idx_of "${MODEL}" "${MODELS[@]}" )"
[[ -n "${EVAL_ABLATION:-}" ]] && ABL_I="$( idx_of "${ABLATION}" "${ABLATIONS[@]}" )"
if [[ -n "${EVAL_BATCH:-}" ]]; then
  [[ "${BATCH}" =~ ^[123]$ ]] || { echo "[FATAL] EVAL_BATCH must be 1|2|3, got '${BATCH}'" >&2; exit 2; }
  BATCH_I=$(( BATCH - 1 ))
fi

GEN_ROOT="${GEN_ROOT:-${OUTPUT_ROOT}/generation}"
GENERATED_DIR="${GEN_ROOT}/${ABLATION}/batch${BATCH}"
ARTIFACT="${GENERATED_DIR}/${MODEL}/artifacts/${MODEL}_seq${SEQ}.pt"
RESULTS_DIR="${EVAL_RESULTS_DIR:-${GEN_ROOT}/${ABLATION}/evaluation_batch${BATCH}/${MODEL}_seq${SEQ}}"
SUMMARY="${RESULTS_DIR}/complete_evaluation.json"

# Preview the cell a given array index resolves to (validation/debug):
#   EVAL_DRY_RUN=1 SLURM_ARRAY_TASK_ID=431 bash scripts/mini_exp/eval_gen_batches.sh
if [[ -n "${EVAL_DRY_RUN:-}" ]]; then
  echo "DRY_RUN tid=${TID} ablation=${ABLATION} batch=${BATCH} model=${MODEL} seq=${SEQ}"
  echo "DRY_RUN artifact=${ARTIFACT} results=${RESULTS_DIR}"
  exit 0
fi

if [[ ! -f "${ARTIFACT}" ]]; then
  echo "[FATAL] artifact missing: ${ARTIFACT}" >&2
  exit 3
fi

# A cell is complete only when its summary holds exactly its own key with a
# non-empty payload; the evaluator logs per-artifact failures and still exits
# 0, so rc alone is not enough.
check_summary() {
  python - "$1" "$2" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
try:
    data = json.loads(open(path).read())
except Exception as exc:  # unreadable/truncated summary
    raise SystemExit(f"[bad] {path}: {exc}")
if list(data) != [key] or not data[key]:
    raise SystemExit(f"[bad] {path}: keys={list(data)}")
print(f"[ok] {key}: {len(data[key])} metric blocks")
PY
}

# Idempotent re-submission: a finished cell is skipped instead of recomputed, so
# an array can be re-submitted to fill holes without redoing work. EVAL_FORCE=1
# forces the evaluation (used when the artifacts change).
if [[ -z "${EVAL_FORCE:-}" && -f "${SUMMARY}" ]] \
   && check_summary "${SUMMARY}" "${MODEL}_seq${SEQ}" >/dev/null 2>&1; then
  echo "=== SKIP ${MODEL}_seq${SEQ}: complete summary already present $(date -Is) ==="
  exit 0
fi

echo "=== GENEVAL cell=${MODEL}_seq${SEQ} ablation=${ABLATION} batch=${BATCH} tid=${TID} ==="
echo "=== slots=$(nproc) OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OMP_WAIT_POLICY=${OMP_WAIT_POLICY} host=$(hostname) ==="
echo "=== artifact=${ARTIFACT}"
echo "=== results=${RESULTS_DIR} start=$(date -Is) ==="

RC=0
if [[ -n "${EVAL_TIME_PHASES:-}" ]]; then
  # Diagnostic mode: timestamp every output line so the cost of each phase is
  # visible (the evaluator prints phase markers, but its stdout is block
  # buffered, so without this the log is silent for tens of minutes at a time).
  set +e
  python -u -m src.unified_evaluator \
    --generated_dir "${GENERATED_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --model "${MODEL}" \
    --seq_length "${SEQ}" \
    --skip_regenerate 2>&1 | while IFS= read -r line; do
      printf '%s %s\n' "$(date +%H:%M:%S)" "${line}"
    done
  RC="${PIPESTATUS[0]}"
  set -e
else
  python -m src.unified_evaluator \
    --generated_dir "${GENERATED_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --model "${MODEL}" \
    --seq_length "${SEQ}" \
    --skip_regenerate || RC=$?
fi
echo "=== evaluator rc=${RC} $(date -Is) ==="
if [[ "${RC}" -ne 0 ]]; then
  echo "=== GENEVAL FAILED cell=${MODEL}_seq${SEQ} rc=${RC} $(date -Is) ==="
  exit "${RC}"
fi

check_summary "${SUMMARY}" "${MODEL}_seq${SEQ}" || exit 4

# Plots for this cell: part of "end to end", but never fatal — a plotting bug
# must not fail an evaluation that already produced all its metrics.
if ! python -m src.plot_statistics.evaluation_plots \
      --results_dir "${RESULTS_DIR}" --output_dir "${RESULTS_DIR}/plots"; then
  echo "[WARN] plot generation failed for ${MODEL}_seq${SEQ} (metrics are complete)"
fi

echo "=== GENEVAL DONE cell=${MODEL}_seq${SEQ} rc=0 $(date -Is) ==="
