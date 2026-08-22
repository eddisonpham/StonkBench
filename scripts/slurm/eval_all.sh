#!/bin/bash
#SBATCH --job-name=sb-eval-all
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/eval_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/eval_%j.err
# ---------------------------------------------------------------------------
# StonkBench — Full Evaluation + Downstream Tasks + Plot Generation
#
# Usage (via sbatch):
#   SEQ_LENGTHS="21 42 126 252" \
#   GENERATED_DIR="outputs/results/latest" \
#   RESULTS_DIR="outputs/results/latest/evaluation" \
#   PLOTS_DIR="outputs/results/latest/plots" \
#   sbatch scripts/slurm/eval_all.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/slurm/common.sh"

SEQ_LENGTHS="${SEQ_LENGTHS:-21 42 126 252}"
GENERATED_DIR="${GENERATED_DIR:-${OUTPUT_ROOT}/results/${STONKBENCH_RUN_ID}}"
RESULTS_DIR="${RESULTS_DIR:-${GENERATED_DIR}/evaluation}"
PLOTS_DIR="${PLOTS_DIR:-${GENERATED_DIR}/plots}"
SKIP_REGENERATE="${SKIP_REGENERATE:-1}"

cd "${PROJECT_ROOT}"

echo "================================================"
echo "StonkBench — Full Evaluation Pipeline"
echo "================================================"
echo "Run ID:       ${STONKBENCH_RUN_ID}"
echo "Seq lengths:  ${SEQ_LENGTHS}"
echo "Generated:    ${GENERATED_DIR}"
echo "Results:      ${RESULTS_DIR}"
echo "Plots:        ${PLOTS_DIR}"
echo "================================================"

# ─── Phase 1: Run evaluations ──────────────────────────────────────────
echo ""
echo "[Phase 1] Running unified evaluator..."
python -m src.unified_evaluator \
  --generated_dir "${GENERATED_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --seq_lengths ${SEQ_LENGTHS} \
  --skip_regenerate

# ─── Phase 2: Generate all plots ───────────────────────────────────────
echo ""
echo "[Phase 2] Generating evaluation plots..."
python -m src.plot_statistics.evaluation_plots \
  --results_dir "${RESULTS_DIR}" \
  --output_dir "${PLOTS_DIR}"

echo ""
echo "================================================"
echo "Evaluation complete!"
echo "Results: ${RESULTS_DIR}"
echo "Plots:   ${PLOTS_DIR}"
echo "================================================"
