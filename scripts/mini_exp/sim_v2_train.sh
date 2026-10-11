#!/bin/bash
#SBATCH --job-name=simv2
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-27
#SBATCH --output=/home/%u/StonkBench/slurm_logs/simv2_%A_%a.out
#SBATCH --error=/home/%u/StonkBench/slurm_logs/simv2_%A_%a.err

# NATIVE per-sequence-length retraining (simulation_v2).
#
# fair_20261004 trained every model on W252 windows and then sliced the
# generated 252-tensor down to seq 21/42/126 (adapters without
# supports_arbitrary_generation generate at batch.inferred_length and the
# pipeline calls stitch_sequences). Here every (model, seq_len) pair gets
# its OWN training run on dl_set_W{seq_len} with
# generation_length = seq_len = the single entry of --seq_lengths, so
# batch.inferred_length == seq_len for every adapter and the stitch branch
# is never reached (verified: stitch only fires when
# generated.data.shape[1] != seq_len).
#
# One array task = one (model, len) training, fully parallel across the
# compute partition. Outputs: $OUTPUT_ROOT/results/simulation_v2/,
# sanity/simulation_v2/, logs/simulation_v2/ (isolated run id).
#
# Submit from trig-login01:
#   command sbatch --export=ALL,STONKBENCH_RUN_ID=simulation_v2,NUM_SAMPLES=500 \
#     scripts/mini_exp/sim_v2_train.sh
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

MODELS=(unconditional_tsdiffusion conditional_tsdiffusion quantgan timegrad vrnn_klsched kalman_vae_safe cond_sig_wgan)
LENS=(21 42 126 252)
TID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set (array 0-27)}"
MODEL="${MODELS[$(( TID / 4 ))]}"
LEN="${LENS[$(( TID % 4 ))]}"
RUN_ID="${STONKBENCH_RUN_ID:?STONKBENCH_RUN_ID must be set (e.g. simulation_v2)}"

# AFTER common.sh: common.sh hardcodes dl_set.pt (the W252 symlink), so the
# native-window override has to happen here or it is a silent no-op
# (same trap documented in scripts/slurm/final_training.sh).
export STONKBENCH_DL_SET_PATH="${PROJECT_ROOT}/data/preprocessed/dl_set_W${LEN}.pt"
if [[ ! -f "${STONKBENCH_DL_SET_PATH}" ]]; then
  echo "[FATAL] dl_set missing: ${STONKBENCH_DL_SET_PATH}"
  exit 2
fi

echo "=== SIMV2 TRAIN model=${MODEL} len=${LEN} run=${RUN_ID} dl_set=$(basename "${STONKBENCH_DL_SET_PATH}") $(date -Is) ==="
RC=0
python -m src.experiments.run_final_training \
  --output_root "${OUTPUT_ROOT}" \
  --run_id "${RUN_ID}" \
  --generation_length "${LEN}" \
  --seq_lengths "${LEN}" \
  --num_samples "${NUM_SAMPLES:-500}" \
  --seed "${SEED:-42}" \
  --device "${STONKBENCH_DEVICE:-cuda}" \
  --models "${MODEL}" || RC=$?
echo "=== SIMV2 DONE model=${MODEL} len=${LEN} rc=${RC} $(date -Is) ==="
exit "${RC}"
