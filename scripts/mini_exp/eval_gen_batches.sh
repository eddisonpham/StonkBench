#!/bin/bash
#SBATCH --job-name=geneval
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/geneval_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/geneval_%A_%a.err

# ============================================================================
# STAGED — DO NOT SUBMIT YET.
# Taxonomy + downstream evaluation over the multi-batch generation trees.
# Prepared as part of the full-scale ablation plan; the user asked for the
# evals to be staged but NOT run yet. Submit only on explicit request:
#   command sbatch --export=ALL scripts/mini_exp/eval_gen_batches.sh
# ============================================================================

# One array task = one (ablation, batch) tree:
#   tid/3 -> ablation, tid%3 -> batch (1..3)
#   generated_dir = ${GEN_ROOT}/<ablation>/batch<k>
#   results_dir   = ${GEN_ROOT}/<ablation>/evaluation_batch<k}
#
# Each task writes its OWN results_dir, so the 9 parallel writers never race
# on complete_evaluation.json (the evaluator OVERWRITES that file per run —
# see the eval_cell.sh note).
#
# NO --seq_lengths filter: every artifact is already stored at its native
# sequence_length (v2 cells are native; fair/latest cells are pre-trimmed to
# L), so the evaluator scores each artifact at its own length. Passing
# --seq_lengths would trim the LAST L steps instead.
#
# Real-data side: common.sh pins STONKBENCH_DL_SET_PATH to data/preprocessed/
# dl_set.pt (the W252 symlink) — identical to the simulation_v2 eval protocol
# that produced the accepted PASS=19 / WARN=9 scorecard, so results stay
# comparable.
#
# Optional single-cell parallelisation (mirrors eval_cell.sh):
#   EVAL_MODEL=timegrad EVAL_SEQ=126  -> --model/--seq_length filters
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

ABLATIONS=(simulation_v2 fair_20261004 latest)
TID="${SLURM_ARRAY_TASK_ID:-0}"
ABLATION="${EVAL_ABLATION:-${ABLATIONS[$(( TID / 3 ))]}}"
BATCH="${EVAL_BATCH:-$(( (TID % 3) + 1 ))}"

GEN_ROOT="${GEN_ROOT:-${OUTPUT_ROOT}/generation}"
GENERATED_DIR="${GEN_ROOT}/${ABLATION}/batch${BATCH}"
RESULTS_DIR="${GEN_ROOT}/${ABLATION}/evaluation_batch${BATCH}"

if [[ ! -d "${GENERATED_DIR}" ]]; then
  echo "[FATAL] generated tree missing: ${GENERATED_DIR}"
  exit 2
fi

FILTERS=()
[[ -n "${EVAL_MODEL:-}" ]] && FILTERS+=(--model "${EVAL_MODEL}")
[[ -n "${EVAL_SEQ:-}" ]] && FILTERS+=(--seq_length "${EVAL_SEQ}")

echo "=== GENEVAL ablation=${ABLATION} batch=${BATCH} generated=${GENERATED_DIR} results=${RESULTS_DIR} filters=${FILTERS[*]:-none} $(date -Is) ==="
RC=0
python -m src.unified_evaluator \
  --generated_dir "${GENERATED_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  ${FILTERS[@]+"${FILTERS[@]}"} || RC=$?
echo "=== GENEVAL DONE ablation=${ABLATION} batch=${BATCH} rc=${RC} $(date -Is) ==="
exit "${RC}"
