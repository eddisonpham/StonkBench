#!/bin/bash
#SBATCH --job-name=genstat
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-59
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/genstat_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/genstat_%A_%a.err

# Multi-batch generation for the 5 STATISTICAL baselines so every ablation tree
# is complete for the evaluative stage (block_bootstrap,
# stationary_block_bootstrap, merton_jump_diffusion, de_jump_diffusion, garch11).
#
# These adapters have no checkpoints: generate_sequences.py re-fits them on the
# stats_set train split inside the job (sub-second) and samples at the native
# length. Artifacts stay in RAW space (no denormalization), matching
# run_model_experiment, and carry model_type="statistical" +
# fit_from_stats_set=true so the evaluator routes them to the statistical
# real-data reference (stats_set test windows).
#
# array map: tid/20 -> ablation, (tid%20)/4 -> model, tid%4 -> seq_len
# Submit from trig-login01:
#   command sbatch --export=ALL scripts/mini_exp/gen_batches_stat.sh
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

ABLATIONS=(simulation_v2 fair_20261004 latest)
MODELS=(block_bootstrap stationary_block_bootstrap merton_jump_diffusion de_jump_diffusion garch11)
LENS=(21 42 126 252)
TID="${SLURM_ARRAY_TASK_ID:-0}"

ABLATION="${GEN_ABLATION:-${ABLATIONS[$(( TID / 20 ))]}}"
MODEL="${GEN_MODEL:-${MODELS[$(( (TID % 20 ) / 4 ))]}}"
LEN="${GEN_SEQ:-${LENS[$(( TID % 4 ))]}}"
BATCHES="${BATCHES:-3}"
SAMPLES="${GEN_SAMPLES:-500}"
OUT_ROOT="${GEN_OUT_ROOT:-${OUTPUT_ROOT}/generation}"

export STONKBENCH_DEVICE="${STONKBENCH_DEVICE:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "=== GENSTAT ablation=${ABLATION} model=${MODEL} len=${LEN} batches=${BATCHES} samples=${SAMPLES} out=${OUT_ROOT} $(date -Is) ==="
RC=0
"${HOME}/.venvs/stonkbench/bin/python" scripts/mini_exp/generate_sequences.py \
  --ablation "${ABLATION}" \
  --model "${MODEL}" \
  --seq "${LEN}" \
  --batches "${BATCHES}" \
  --samples "${SAMPLES}" \
  --seed-base "${GEN_SEED_BASE:-42}" \
  --out-root "${OUT_ROOT}" || RC=$?
echo "=== GENSTAT DONE ablation=${ABLATION} model=${MODEL} len=${LEN} rc=${RC} $(date -Is) ==="
exit "${RC}"
