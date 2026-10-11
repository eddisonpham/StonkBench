#!/bin/bash
#SBATCH --job-name=evalfinal
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=/home/%u/StonkBench/slurm_logs/evalfinal_%j.out
#SBATCH --error=/home/%u/StonkBench/slurm_logs/evalfinal_%j.err

# Post-evaluation finalizer. Submit it chained to the eval arrays so the
# aggregation happens as soon as they end, with no session needed:
#
#   command sbatch --export=ALL --dependency=afterany:<evaljobid> \
#     scripts/mini_exp/finalize_eval.sh
#
# It reports cell coverage for the whole grid and, ONLY when every expected cell
# is present and well-formed, runs the batch aggregation. It never resubmits
# evaluation work: if cells are missing it says which ones and exits non-zero,
# leaving the decision to a human (an earlier auto-resubmitting watcher of mine
# piled up duplicate array jobs, so that is deliberately not automated).
#
# Nothing here scales scientific time: this job waits for the evals, spends
# seconds in the aggregator, and is cheap enough to free immediately.
set -uo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

OUT="${GEN_ROOT:-${OUTPUT_ROOT}/generation}"
ABLATIONS=(simulation_v2 fair_20261004 latest)
REPORT="${OUT}/eval_finalize_report.txt"

exec > >(tee "${REPORT}") 2>&1
echo "=============================================================="
echo "eval finalize — $(date -Is)  host=$(hostname)"
echo "root: ${OUT}"
echo "=============================================================="

python scripts/mini_exp/check_eval_coverage.py --root "${OUT}" \
  --ablation "${ABLATIONS[@]}"
COVERAGE_RC=$?
echo "coverage rc=${COVERAGE_RC}"

if [[ "${COVERAGE_RC}" -ne 0 ]]; then
  echo "[finalize] NOT aggregating: the grid is incomplete above."
  echo "[finalize] top up the missing cells with, e.g.:"
  echo "  # after computing the missing tids, one short array per contiguous run"
  echo "  command sbatch --export=ALL,GEN_ROOT=${OUT} --time=02:00:00 \\"
  echo "    --array=<missing tids> scripts/mini_exp/eval_gen_batches.sh"
  exit 1
fi

echo
echo "[finalize] aggregating mean ± std across the 3 replicate batches"
python scripts/mini_exp/aggregate_batch_metrics.py --root "${OUT}" \
  --ablation "${ABLATIONS[@]}"
AGG_RC=$?
echo "aggregate rc=${AGG_RC}"

echo
echo "[finalize] outputs:"
ls -1 "${OUT}"/*/batch_metrics_aggregate.json 2>/dev/null || true
echo "[finalize] done $(date -Is)"
exit "${AGG_RC}"
