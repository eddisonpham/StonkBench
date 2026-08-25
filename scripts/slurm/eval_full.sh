#!/bin/bash
#SBATCH --job-name=sb-eval-full
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/eval_full_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/eval_full_%j.err

# Full evaluation of all models at all sequence lengths (21, 42, 126, 252)
# followed by downstream task evaluation and plot generation.
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

SEQ_LENGTHS="21 42 126 252"
GENERATED_DIR="${OUTPUT_ROOT}/results/${STONKBENCH_RUN_ID}"
RESULTS_DIR="${GENERATED_DIR}/evaluation"
PLOTS_DIR="${GENERATED_DIR}/plots"

echo "================================================"
echo "StonkBench — Full Evaluation Pipeline"
echo "================================================"
echo "Run ID:       ${STONKBENCH_RUN_ID}"
echo "Seq lengths:  ${SEQ_LENGTHS}"
echo "Generated:    ${GENERATED_DIR}"
echo "Results:      ${RESULTS_DIR}"
echo "Plots:        ${PLOTS_DIR}"
echo "Models present:"
ls -d "${GENERATED_DIR}"/*/artifacts/ 2>/dev/null | sed 's|.*/latest/||;s|/artifacts/||' | sort
echo "================================================"

# ─── Phase 1: Run evaluations ──────────────────────────────────
echo ""
echo "[Phase 1] Running unified evaluator on all artifacts..."
python -m src.unified_evaluator \
  --generated_dir "${GENERATED_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --seq_lengths ${SEQ_LENGTHS} \
  --skip_regenerate

# ─── Phase 2: Generate all plots ───────────────────────────────
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
