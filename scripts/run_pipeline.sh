#!/bin/bash
# End-to-end StonkBench pipeline on Trillium GPU login (trig-login01):
#   HP search -> aggregate -> final training -> eval -> sync to home
#
# Usage (from trig-login01):
#   bash scripts/run_pipeline.sh --smoke --submit-only   # short GPU sanity pipeline
#   bash scripts/run_pipeline.sh --submit-only            # full GPU pipeline
#   bash scripts/run_pipeline.sh sync                    # sync scratch -> home
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
SCRATCH_ROOT="${SCRATCH:-/scratch/$USER}"
STAGING_ROOT="${SCRATCH_ROOT}/stonkbench/output"
CANONICAL_OUTPUT="/home/epham/StonkBench/output"
SLURM_DIR="${PROJECT_ROOT}/scripts/slurm"
SUBMIT_ONLY=0
SMOKE=0

require_gpu_login() {
  local host
  host="$(hostname)"
  if [[ "${host}" =~ ^trig ]]; then
    return 0
  fi

  # CPU login (tri-login*): forward to GPU login instead of failing immediately.
  local fwd_args=()
  [[ "${SMOKE}" == "1" ]] && fwd_args+=(--smoke)
  [[ "${SUBMIT_ONLY}" == "1" ]] && fwd_args+=(--submit-only)
  echo "On CPU login (${host}); forwarding submission to trig-login01..."
  exec ssh -o BatchMode=yes -o ConnectTimeout=15 trig-login01 \
    "cd $(printf '%q' "${PROJECT_ROOT}") && bash scripts/run_pipeline.sh $(printf '%q ' "${fwd_args[@]}")"
}

sync_to_home() {
  if [[ ! -d "${STAGING_ROOT}" ]]; then
    echo "No staged outputs at ${STAGING_ROOT}"
    exit 1
  fi

  mkdir -p "${CANONICAL_OUTPUT}"
  echo "Syncing ${STAGING_ROOT} -> ${CANONICAL_OUTPUT}"
  rsync -av "${STAGING_ROOT}/" "${CANONICAL_OUTPUT}/"

  echo ""
  echo "Output summary:"
  for subdir in checkpoints experiments logs results sanity; do
    count=$(find "${CANONICAL_OUTPUT}/${subdir}" -type f 2>/dev/null | wc -l)
    echo "  ${subdir}/  (${count} files)"
  done

  local artifacts checkpoints sanity_pngs
  artifacts=$(find "${CANONICAL_OUTPUT}/results" -path '*/artifacts/*.pt' 2>/dev/null | wc -l)
  checkpoints=$(find "${CANONICAL_OUTPUT}/checkpoints" -type f 2>/dev/null | wc -l)
  sanity_pngs=$(find "${CANONICAL_OUTPUT}/sanity" -name '*.png' 2>/dev/null | wc -l)

  if [[ "${artifacts}" -eq 0 ]] || [[ "${checkpoints}" -eq 0 ]] || [[ "${sanity_pngs}" -eq 0 ]]; then
    echo ""
    echo "WARNING: final training outputs look incomplete:"
    echo "  model artifacts (.pt): ${artifacts}"
    echo "  checkpoints:           ${checkpoints}"
    echo "  sanity plots (.png):   ${sanity_pngs}"
    return 1
  fi
  echo "Sync complete (artifacts, checkpoints, sanity plots present)."
}

wait_for_array() {
  local job_id="$1"
  local label="$2"
  echo "Waiting for ${label} (array ${job_id})..."
  while squeue -j "${job_id}" -h 2>/dev/null | grep -q .; do
    sleep 60
  done
  echo "${label} array finished (individual tasks may have failed; aggregate uses available trials)"
}

wait_for_job() {
  local job_id="$1"
  local label="$2"
  echo "Waiting for ${label} (job ${job_id})..."
  while squeue -j "${job_id}" -h 2>/dev/null | grep -q .; do
    sleep 60
  done
  local state exit_code
  state=$(sacct -j "${job_id}" --format=State --noheader -P 2>/dev/null | head -1 | cut -d'|' -f1)
  exit_code=$(sacct -j "${job_id}" --format=ExitCode --noheader -P 2>/dev/null | head -1 | cut -d'|' -f1)
  echo "${label} finished: state=${state} exit=${exit_code}"
  if [[ "${state}" != "COMPLETED" ]] || [[ "${exit_code}" != "0:0" ]]; then
    echo "ERROR: ${label} did not succeed. Check ${SCRATCH_ROOT}/stonkbench/slurm_logs/"
    exit 1
  fi
}

# Bypass login-node sbatch wrapper (--export=NONE) so smoke/device flags reach jobs.
sbatch_gpu() {
  /opt/slurm/bin/sbatch --export=NONE --get-user-env \
    --export=STONKBENCH_SMOKE="${SMOKE}",STONKBENCH_DEVICE=cuda \
    "$@"
}

submit_pipeline() {
  require_gpu_login
  mkdir -p "${SCRATCH_ROOT}/stonkbench/slurm_logs"
  cd "${PROJECT_ROOT}"

  local hp_id agg_id train_id eval_id

  if [[ "${SMOKE}" == "1" ]]; then
    # Tiny grid on compute (debug QoS allows only 1 submitted job).
    # 8 HP trials + 14 model trains, capped concurrency.
    hp_id=$(sbatch_gpu --parsable \
      --time=00:45:00 --array=0-7%4 \
      "${SLURM_DIR}/hp_search.sh")
    agg_id=$(sbatch_gpu --parsable \
      --time=00:15:00 \
      --dependency=afterany:"${hp_id}" "${SLURM_DIR}/hp_aggregate.sh")
    train_id=$(sbatch_gpu --parsable \
      --time=00:45:00 --array=0-13%4 \
      --dependency=afterok:"${agg_id}" "${SLURM_DIR}/final_training.sh")
    eval_id=$(sbatch_gpu --parsable \
      --time=00:30:00 \
      --dependency=afterany:"${train_id}" "${SLURM_DIR}/eval.sh")

    echo "Submitted SMOKE GPU pipeline:"
    echo "  1 HP search (8 trials, 1 GPU each, max 4): ${hp_id}"
    echo "  2 HP aggregate:                            ${agg_id}  (afterany:${hp_id})"
    echo "  3 Final training (14 models, max 4):       ${train_id}  (afterok:${agg_id})"
    echo "  4 Evaluation:                              ${eval_id}  (afterany:${train_id})"
  else
    # Full grid: 72 HP trials (max 8 concurrent GPUs), 14 trains (max 4).
    hp_id=$(sbatch_gpu --parsable "${SLURM_DIR}/hp_search.sh")
    agg_id=$(sbatch_gpu --parsable \
      --dependency=afterany:"${hp_id}" "${SLURM_DIR}/hp_aggregate.sh")
    train_id=$(sbatch_gpu --parsable \
      --dependency=afterok:"${agg_id}" "${SLURM_DIR}/final_training.sh")
    eval_id=$(sbatch_gpu --parsable \
      --dependency=afterany:"${train_id}" "${SLURM_DIR}/eval.sh")

    echo "Submitted FULL GPU pipeline:"
    echo "  1 HP search (72 trials, 1 GPU each, max 8): ${hp_id}"
    echo "  2 HP aggregate:                             ${agg_id}  (afterany:${hp_id})"
    echo "  3 Final training (14 models, max 4 GPUs):   ${train_id}  (afterok:${agg_id})"
    echo "  4 Evaluation:                               ${eval_id}  (afterany:${train_id})"
  fi

  echo ""
  echo "Staged on compute: ${STAGING_ROOT}"
  echo "Final destination: ${CANONICAL_OUTPUT}"
  echo "Logs: ${SCRATCH_ROOT}/stonkbench/slurm_logs/"

  if [[ "${SUBMIT_ONLY}" == "1" ]]; then
    echo ""
    echo "Jobs submitted. When finished, run: bash scripts/run_pipeline.sh sync"
    return 0
  fi

  wait_for_array "${hp_id}" "HP search"
  wait_for_job "${agg_id}" "HP aggregate"
  wait_for_array "${train_id}" "Final training"
  wait_for_job "${eval_id}" "Evaluation"
  sync_to_home
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    sync)
      sync_to_home
      exit 0
      ;;
    --submit-only)
      SUBMIT_ONLY=1
      shift
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    run)
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/run_pipeline.sh [--smoke] [--submit-only | sync]"
      exit 0
      ;;
    *)
      echo "Usage: bash scripts/run_pipeline.sh [--smoke] [--submit-only | sync]"
      exit 1
      ;;
  esac
done

submit_pipeline
