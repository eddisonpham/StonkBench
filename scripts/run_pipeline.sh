#!/bin/bash
# End-to-end StonkBench pipeline (login node):
#   HP search -> aggregate -> final training -> eval -> sync to /home/epham/StonkBench/output
#
# Usage:
#   bash scripts/run_pipeline.sh              # submit, wait, sync
#   bash scripts/run_pipeline.sh --submit-only
#   bash scripts/run_pipeline.sh sync         # sync staged scratch outputs to home only
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
SCRATCH_ROOT="${SCRATCH:-/scratch/$USER}"
STAGING_ROOT="${SCRATCH_ROOT}/stonkbench/output"
CANONICAL_OUTPUT="/home/epham/StonkBench/output"
SLURM_DIR="${PROJECT_ROOT}/scripts/slurm"

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

submit_pipeline() {
  mkdir -p "${SCRATCH_ROOT}/stonkbench/slurm_logs"
  cd "${PROJECT_ROOT}"

  local hp_id agg_id train_id eval_id
  hp_id=$(sbatch --parsable "${SLURM_DIR}/hp_search.sh")
  # afterany: run aggregate even if some HP array tasks fail (timeouts)
  agg_id=$(sbatch --parsable --dependency=afterany:"${hp_id}" "${SLURM_DIR}/hp_aggregate.sh")
  train_id=$(sbatch --parsable --dependency=afterok:"${agg_id}" "${SLURM_DIR}/final_training.sh")
  eval_id=$(sbatch --parsable --dependency=afterany:"${train_id}" "${SLURM_DIR}/eval.sh")

  echo "Submitted pipeline:"
  echo "  1 HP search (6 nodes):  ${hp_id}  (9 trials/node, 6 parallel)"
  echo "  2 HP aggregate:         ${agg_id}  (afterany:${hp_id})"
  echo "  3 Final training (4):   ${train_id}  (3 models/node, afterok:${agg_id})"
  echo "  4 Evaluation:           ${eval_id}  (afterany:${train_id})"
  echo ""
  echo "Staged on compute: ${STAGING_ROOT}"
  echo "Final destination: ${CANONICAL_OUTPUT}"

  if [[ "${SUBMIT_ONLY:-0}" == "1" ]]; then
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

case "${1:-run}" in
  sync)
    sync_to_home
    ;;
  --submit-only)
    SUBMIT_ONLY=1 submit_pipeline
    ;;
  run|"")
    submit_pipeline
    ;;
  *)
    echo "Usage: bash scripts/run_pipeline.sh [--submit-only | sync]"
    exit 1
    ;;
esac
