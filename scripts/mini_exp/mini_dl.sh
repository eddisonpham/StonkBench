#!/bin/bash
#SBATCH --job-name=mini-dl
#SBATCH --account=def-yqhuang
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:55:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/mini_dl_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/mini_dl_%j.err

# Mini DL validation runs (debug partition; qos=debug allows only 1 job per
# user, so this is ONE sequential job over a selectable model batch).
# Short training budget with early stopping disabled so full training curves
# are captured; fidelity vs ground truth logged for seq 21/42/126/252.
# All outputs land in $STONKBENCH_MINI_ROOT/$STONKBENCH_RUN_ID/ — nothing
# touches the production outputs/ tree.
#
# Submit from trig-login01 (GPU login):
#   STONKBENCH_RUN_ID=mini_<tag> MODELS_SEL="quantgan timegrad" \
#     sbatch scripts/mini_exp/mini_dl.sh
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/StonkBench}/scripts/slurm/common.sh"
cd "${PROJECT_ROOT}"

# Short per-model budgets (production wall-clocks in parens):
#   quantgan 529s/51ep, kalman 2747s/27ep, vrnn 1368s/15ep, tsdiff ~40s,
#   timegrad 44s/100ep, sigwgan 607s/1500steps.
declare -A EPOCHS=(
  [quantgan]=30
  [timegrad]=30
  [kalman_vae_safe]=20
  [unconditional_tsdiffusion]=30
  [conditional_tsdiffusion]=30
  [vrnn_klsched]=20
  [cond_sig_wgan]=100            # nominal; step-based (see --steps below)
)

DEFAULT_SEL="timegrad unconditional_tsdiffusion conditional_tsdiffusion quantgan cond_sig_wgan vrnn_klsched kalman_vae_safe"
# Batch groups (env-safe: the trig-login01 `sbatch` wrapper forces
# --export=NONE, and --export values may contain neither spaces nor commas).
#   BATCH=1  cheap models (~30 min): diffusion x2, timegrad, quantgan, sigwgan
#   BATCH=2  heavy models (~80 min): vrnn, kalman
#   BATCH=all (default) everything
# MODELS_SEL (comma/space separated) also works when set in the shell that
# invokes this script directly (not through sbatch --export).
case "${BATCH:-}" in
  1) SELECTED="timegrad unconditional_tsdiffusion conditional_tsdiffusion quantgan cond_sig_wgan" ;;
  2) SELECTED="vrnn_klsched kalman_vae_safe" ;;
  3) SELECTED="cond_sig_wgan vrnn_klsched kalman_vae_safe" ;;
  4) SELECTED="cond_sig_wgan" ;;
  *) SELECTED="${MODELS_SEL:-$DEFAULT_SEL}" ;;
esac
# Comma- or space-separated model list.
SELECTED="${SELECTED//,/ }"

RUN_ID="${STONKBENCH_RUN_ID:?STONKBENCH_RUN_ID must be set (e.g. mini_20261004a)}"
MINI_ROOT="${STONKBENCH_MINI_ROOT:-/scratch/$USER/stonkbench/mini_exp}"
mkdir -p "${MINI_ROOT}/${RUN_ID}/events"

RC=0
for MODEL in ${SELECTED}; do
  # Live training-curve events (env-gated hook in the DL adapters).
  export STONKBENCH_MINI_EVENTS="${MINI_ROOT}/${RUN_ID}/events/${MODEL}.jsonl"
  echo "=== MINI DL ${MODEL} run=${RUN_ID} epochs=${EPOCHS[$MODEL]} $(date -Is) ==="
  if ! python scripts/mini_exp/run_mini.py \
      --model "${MODEL}" \
      --run_id "${RUN_ID}" \
      --out_root "${MINI_ROOT}" \
      --epochs "${EPOCHS[$MODEL]}" \
      --num_samples "${NUM_SAMPLES:-200}" \
      --seq_lengths "${SEQ_LENGTHS:-21,42,126,252}" \
      --device "${STONKBENCH_DEVICE:-cuda}" \
      --seed "${SEED:-42}"; then
    echo "=== FAILED ${MODEL} $(date -Is) ==="
    RC=1
    # keep going: partial batches still produce usable logs
  fi
  echo "=== DONE ${MODEL} $(date -Is) ==="
done
exit ${RC}
