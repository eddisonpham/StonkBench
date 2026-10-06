#!/bin/bash
#SBATCH --job-name=simv2-recheck
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/simeval_cell_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/simeval_cell_%j.err

# Re-evaluate ONE (model, seq_length) cell of simulation_v2.
#
# Why a separate results_dir: unified_evaluator.run() OVERWRITES
# <results_dir>/complete_evaluation.json with only the artifacts evaluated
# in that invocation, so a filtered run in the main evaluation tree would
# clobber the 28-cell summary. This job writes into evaluation_recheck/;
# afterwards the seq_L/<model>/ outputs and the single complete_evaluation
# entry are spliced into the main tree (see MODEL_PAPER_AUDIT.md §10).
#
# Usage (from trig-login01):
#   command sbatch --export=ALL,CELL_MODEL=timegrad,CELL_SEQ=126 \
#     scripts/mini_exp/eval_cell.sh
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

: "${CELL_MODEL:?CELL_MODEL env var required (e.g. timegrad)}"
: "${CELL_SEQ:?CELL_SEQ env var required (e.g. 126)}"

python -m src.unified_evaluator \
  --generated_dir "${OUTPUT_ROOT}/results/simulation_v2" \
  --results_dir "${OUTPUT_ROOT}/../mini_exp/simulation_v2/evaluation_recheck" \
  --model "${CELL_MODEL}" \
  --seq_length "${CELL_SEQ}"

echo "=== CELL RECHECK ${CELL_MODEL}@${CELL_SEQ} done rc=$? $(date -Is) ==="
