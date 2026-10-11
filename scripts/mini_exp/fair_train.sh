#!/bin/bash
#SBATCH --job-name=fair-train
#SBATCH --account=def-yqhuang
#SBATCH --partition=compute_b200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/StonkBench/slurm_logs/fair_train_%j.out
#SBATCH --error=/home/%u/StonkBench/slurm_logs/fair_train_%j.err

# FULL-BUDGET fair retraining of the DL models, production path
# (run_final_training -> run_model_experiment), with the fairness fix:
#   num_samples=500 for EVERY model (production evaluated 5 DL models at n=9).
# Isolated run id -> results land in $OUTPUT_ROOT/results/$RUN_ID, never in
# results/latest. Live per-epoch events stream to $RUN_DIR/events (env-gated
# emit_epoch hooks). Ends with score_outputs.py = collapse/realism report
# (NOT the downstream evaluation taxonomies).
#
# Groups (debug qos allows only 1 job/user, b200/h200 are usually idle):
#   TRAIN_GROUP=cheap  -> diffusion x2, timegrad, quantgan, cond_sig_wgan
#   TRAIN_GROUP=heavy  -> vrnn_klsched, kalman_vae_safe
#
# Submit from trig-login01, e.g.:
#   command sbatch --export=ALL,TRAIN_GROUP=cheap,STONKBENCH_RUN_ID=fair_20261004 \
#     scripts/mini_exp/fair_train.sh
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${STONKBENCH_RUN_ID:?STONKBENCH_RUN_ID must be set (e.g. fair_20261004)}"
MINI_ROOT="${STONKBENCH_MINI_ROOT:-/home/epham/StonkBench/outputs/mini_exp}"
RUN_DIR="${MINI_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}/events"

# Live training curves (adapter emit_epoch hooks fire when this is set).
export STONKBENCH_MINI_EVENTS="${RUN_DIR}/events/fair_${TRAIN_GROUP:-all}.jsonl"

case "${TRAIN_GROUP:-}" in
  cheap) MODELS="timegrad unconditional_tsdiffusion conditional_tsdiffusion quantgan cond_sig_wgan" ;;
  cheap2) MODELS="unconditional_tsdiffusion conditional_tsdiffusion quantgan cond_sig_wgan" ;;
  sigwgan) MODELS="cond_sig_wgan" ;;
  heavy) MODELS="vrnn_klsched kalman_vae_safe" ;;
  *)     MODELS="timegrad unconditional_tsdiffusion conditional_tsdiffusion quantgan cond_sig_wgan vrnn_klsched kalman_vae_safe" ;;
esac

echo "=== FAIR TRAIN group=${TRAIN_GROUP:-all} run=${RUN_ID} models=${MODELS} $(date -Is) ==="
echo "=== OUTPUT_ROOT=${OUTPUT_ROOT} results=${OUTPUT_ROOT}/results/${RUN_ID} ==="
# One model per process (matches the production array design): vendor path
# bootstraps are process-global, so isolation also protects against any
# cross-vendor shadowing beyond the gluonts fix in training_utils. A failing
# model must not kill the rest of the batch.
TRAIN_RC=0
for M in ${MODELS}; do
  echo "=== TRAIN MODEL ${M} $(date -Is) ==="
  python -m src.experiments.run_final_training \
    --output_root "${OUTPUT_ROOT}" \
    --run_id "${RUN_ID}" \
    --generation_length 252 \
    --seq_lengths 21 42 126 252 \
    --num_samples "${NUM_SAMPLES:-500}" \
    --seed "${SEED:-42}" \
    --device "${STONKBENCH_DEVICE:-cuda}" \
    --models "${M}" || TRAIN_RC=1
done
echo "=== TRAIN DONE rc=${TRAIN_RC} $(date -Is) — scoring outputs for collapse ==="
set +e
python scripts/mini_exp/score_outputs.py \
  --run_id "${RUN_ID}" \
  --output_root "${OUTPUT_ROOT}" \
  --report_dir "${RUN_DIR}"
SCORE_RC=$?
set -e
echo "=== SCORE rc=${SCORE_RC} $(date -Is) ==="

# Rebuild live-curve grid + report.md from the streamed epoch events.
set +e
python scripts/mini_exp/refresh_report.py --run_dir "${RUN_DIR}" > /dev/null 2>&1
REFRESH_RC=$?
set -e
echo "=== REFRESH_REPORT rc=${REFRESH_RC} $(date -Is) ==="
exit $(( TRAIN_RC != 0 ? TRAIN_RC : SCORE_RC ))
