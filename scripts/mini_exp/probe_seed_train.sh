#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-43
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/probe_%A_%a.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/probe_%A_%a.err

# SEED-SWEEP PROBES for simulation_v2 WARN resolution (fairness audit).
#
# Purpose: the pipeline is deterministic given --seed (seq252 artifacts of
# fair_20261004 and simulation_v2 are bit-identical, both seed 42). So the
# only way to measure run-to-run variance of a flagged cell is to retrain
# with a FRESH seed while keeping every other knob (budget, patience,
# num_samples, dl_set, generation_length) EXACTLY as in simulation_v2.
# Nothing is tuned per model: every WARN cell gets the same treatment and a
# handful of PASS cells are probed as controls to estimate baseline variance.
#
# One array task = one (model, len, seed) training. Outputs go to the
# isolated run id probe_seed: $OUTPUT_ROOT/results/probe_seed/.
#
# Submit from trig-login01:
#   command sbatch --export=ALL,NUM_SAMPLES=500 \
#     scripts/mini_exp/probe_seed_train.sh
# (run id is derived per task: probe_s<seed>, so concurrent seeds never
# share an artifact path.)
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

# Explicit auditable task table: MODEL LEN SEED
TASKS=(
# --- WARN cells: seeds 43,44,45 -----------------------------------------
"cond_sig_wgan 21 43"
"cond_sig_wgan 21 44"
"cond_sig_wgan 21 45"
"cond_sig_wgan 42 43"
"cond_sig_wgan 42 44"
"cond_sig_wgan 42 45"
"kalman_vae_safe 21 43"
"kalman_vae_safe 21 44"
"kalman_vae_safe 21 45"
"kalman_vae_safe 42 43"
"kalman_vae_safe 42 44"
"kalman_vae_safe 42 45"
"timegrad 21 43"
"timegrad 21 44"
"timegrad 21 45"
"timegrad 42 43"
"timegrad 42 44"
"timegrad 42 45"
"timegrad 126 43"
"timegrad 126 44"
"timegrad 126 45"
"timegrad 252 43"
"timegrad 252 44"
"timegrad 252 45"
"unconditional_tsdiffusion 252 43"
"unconditional_tsdiffusion 252 44"
"unconditional_tsdiffusion 252 45"
"vrnn_klsched 252 43"
"vrnn_klsched 252 44"
"vrnn_klsched 252 45"
"quantgan 126 43"
"quantgan 126 44"
"quantgan 252 43"
"quantgan 252 44"
# --- PASS-cell controls: seeds 43,44 ------------------------------------
"cond_sig_wgan 126 43"
"cond_sig_wgan 126 44"
"quantgan 42 43"
"quantgan 42 44"
"kalman_vae_safe 126 43"
"kalman_vae_safe 126 44"
"vrnn_klsched 126 43"
"vrnn_klsched 126 44"
"conditional_tsdiffusion 126 43"
"conditional_tsdiffusion 126 44"
)

TID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set (array 0-43)}"
LINE="${TASKS[$(( TID ))]:-}"
[[ -n "$LINE" ]] || { echo "[FATAL] no task for tid=${TID}"; exit 2; }
read -r MODEL LEN SEED <<< "$LINE"
# Per-SEED run id: all seeds of one (model, len) would otherwise share
# results/<run>/<model>/artifacts/<model>_seq<LEN>.pt, and concurrent array
# tasks race on that path (last writer wins, earlier seeds lost). One run id
# per seed => one writer per path, and score_outputs can be looped per seed.
RUN_ID="probe_s${SEED}"

# AFTER common.sh: common.sh hardcodes dl_set.pt (the W252 symlink), so the
# native-window override has to happen here or it is a silent no-op.
export STONKBENCH_DL_SET_PATH="${PROJECT_ROOT}/data/preprocessed/dl_set_W${LEN}.pt"
if [[ ! -f "${STONKBENCH_DL_SET_PATH}" ]]; then
  echo "[FATAL] dl_set missing: ${STONKBENCH_DL_SET_PATH}"
  exit 2
fi

echo "=== PROBE TRAIN model=${MODEL} len=${LEN} seed=${SEED} run=${RUN_ID} dl_set=$(basename "${STONKBENCH_DL_SET_PATH}") $(date -Is) ==="
RC=0
python -m src.experiments.run_final_training \
  --output_root "${OUTPUT_ROOT}" \
  --run_id "${RUN_ID}" \
  --generation_length "${LEN}" \
  --seq_lengths "${LEN}" \
  --num_samples "${NUM_SAMPLES:-500}" \
  --seed "${SEED}" \
  --device "${STONKBENCH_DEVICE:-cuda}" \
  --models "${MODEL}" || RC=$?
echo "=== PROBE DONE model=${MODEL} len=${LEN} seed=${SEED} rc=${RC} $(date -Is) ==="
exit "${RC}"
