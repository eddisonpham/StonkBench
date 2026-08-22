#!/bin/bash
#SBATCH --job-name=sb-eval
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-47
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/eval_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/eval_%A_%a.err

# 48 parallel evaluation tasks: 12 models × 4 seq_lengths.
# Task ID maps to: model_idx = TASK_ID // 4, seq_idx = TASK_ID % 4
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

MODELS=(
  quantgan
  timegrad
  kalman_vae_safe
  unconditional_tsdiffusion
  conditional_tsdiffusion
  vrnn_klsched
  cond_sig_wgan
  block_bootstrap
  stationary_block_bootstrap
  merton_jump_diffusion
  de_jump_diffusion
  garch11
)

SEQ_LENGTHS=(21 42 126 252)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
MODEL_IDX=$(( TASK_ID / 4 ))
SEQ_IDX=$(( TASK_ID % 4 ))

MODEL="${MODELS[${MODEL_IDX}]}"
SEQ="${SEQ_LENGTHS[${SEQ_IDX}]}"

GENERATED_DIR="${OUTPUT_ROOT}/results/${STONKBENCH_RUN_ID}"
RESULTS_DIR="${GENERATED_DIR}/evaluation"

echo "=== EVAL [${TASK_ID}/47] model=${MODEL} seq=${SEQ} $(date -Is) ==="

python -m src.unified_evaluator \
  --generated_dir "${GENERATED_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --model "${MODEL}" \
  --seq_length "${SEQ}" \
  --skip_regenerate

echo "=== DONE [${TASK_ID}/47] ${MODEL} seq${SEQ} $(date -Is) ==="
