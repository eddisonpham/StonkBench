#!/bin/bash
#SBATCH --job-name=gen3
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-83
#SBATCH --output=/home/%u/StonkBench/slurm_logs/gen_%A_%a.out
#SBATCH --error=/home/%u/StonkBench/slurm_logs/gen_%A_%a.err

# Multi-batch sequence generation for the full-scale ablation evaluation.
#
# One array task = one (ablation, model, seq_len) cell = 3 batches x 500
# samples drawn from the in-repo checkpoint via adapter.load_state()
# (checkpoints/<ablation>/<model>/<model>_seq{L}_final.pt). Outputs:
#
#   ${OUT_ROOT}/<ablation>/batch{1,2,3}/<model>/artifacts/<model>_seq{L}.pt
#   ${OUT_ROOT}/<ablation>/batch{1,2,3}/<model>/gen_seq{L}.json
#   ${OUT_ROOT}/<ablation>/ground_truth/ground_truth_seq{L}.pt   (once per ablation)
#
# OUT_ROOT defaults to ${OUTPUT_ROOT}/generation — common.sh resolves
# OUTPUT_ROOT to <repo>/outputs when /home is writable from the compute node
# and to the scratch staging tree otherwise (see the symlink-after-run note in
# common.sh). Override with GEN_OUT_ROOT.
#
# fair_20261004 and latest only have seq-252 checkpoints, so those cells draw
# 252 steps and trim to the requested length (pipeline stitch_sequences =
# prefix slice); the driver asserts this.
#
# array map: tid/28 -> ablation, (tid%28)/4 -> model, tid%4 -> seq_len
# Submit from trig-login01:
#   command sbatch --export=ALL scripts/mini_exp/gen_batches.sh
# Pilot one cell (overrides the array map):
#   command sbatch --export=ALL,GEN_ABLATION=simulation_v2,GEN_MODEL=timegrad,GEN_SEQ=252 \
#     --array=0 scripts/mini_exp/gen_batches.sh
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

ABLATIONS=(simulation_v2 fair_20261004 latest)
MODELS=(unconditional_tsdiffusion conditional_tsdiffusion quantgan timegrad vrnn_klsched kalman_vae_safe cond_sig_wgan)
LENS=(21 42 126 252)
TID="${SLURM_ARRAY_TASK_ID:-0}"

ABLATION="${GEN_ABLATION:-${ABLATIONS[$(( TID / 28 ))]}}"
MODEL="${GEN_MODEL:-${MODELS[$(( (TID % 28) / 4 ))]}}"
LEN="${GEN_SEQ:-${LENS[$(( TID % 4 ))]}}"
BATCHES="${BATCHES:-3}"
SAMPLES="${GEN_SAMPLES:-500}"
OUT_ROOT="${GEN_OUT_ROOT:-${OUTPUT_ROOT}/generation}"

# dl_set is re-pinned per NATIVE length inside generate_sequences.py (AFTER
# common.sh hardcodes the W252 dl_set.pt symlink — same trap documented in
# sim_v2_train.sh).
export STONKBENCH_DEVICE="${STONKBENCH_DEVICE:-cuda}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ ! -f "${HOME}/.venvs/stonkbench/bin/python" ]]; then
  echo "[FATAL] venv python missing"
  exit 2
fi

echo "=== GEN ablation=${ABLATION} model=${MODEL} len=${LEN} batches=${BATCHES} samples=${SAMPLES} out=${OUT_ROOT} device=${STONKBENCH_DEVICE} $(date -Is) ==="
RC=0
"${HOME}/.venvs/stonkbench/bin/python" scripts/mini_exp/generate_sequences.py \
  --ablation "${ABLATION}" \
  --model "${MODEL}" \
  --seq "${LEN}" \
  --batches "${BATCHES}" \
  --samples "${SAMPLES}" \
  --seed-base "${GEN_SEED_BASE:-42}" \
  --out-root "${OUT_ROOT}" || RC=$?
echo "=== GEN DONE ablation=${ABLATION} model=${MODEL} len=${LEN} rc=${RC} $(date -Is) ==="
exit "${RC}"
