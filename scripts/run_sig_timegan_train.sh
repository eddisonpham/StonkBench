#!/bin/bash
# Targeted paper train for TimeGAN + PCF-GAN + SigWGAN only (no stats, no other DL):
#   1) HP sweep (27 trials = 3 models x 9 configs)
#   2) HP aggregate
#   3) Final DL train (3 models, gen_length=100)
#   4) Sync scratch -> home dated results (no --delete of sibling runs)
#
# Usage:
#   STONKBENCH_RUN_ID=2026-07-12_sig_timegan bash scripts/run_sig_timegan_train.sh --submit-only
#   STONKBENCH_RUN_ID=2026-07-12_sig_timegan bash scripts/run_sig_timegan_train.sh sync
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
SCRATCH_ROOT="${SCRATCH:-/scratch/$USER}"
STAGING_ROOT="${SCRATCH_ROOT}/stonkbench/output"
CANONICAL_OUTPUT="/home/epham/StonkBench/output"
SLURM_DIR="${PROJECT_ROOT}/scripts/slurm"
SUBMIT_ONLY=0
GENERATION_LENGTH="${GENERATION_LENGTH:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
STONKBENCH_RUN_ID="${STONKBENCH_RUN_ID:-$(date +%Y-%m-%d)_sig_timegan}"
export STONKBENCH_RUN_ID

require_gpu_login() {
  local host
  host="$(hostname)"
  if [[ "${host}" =~ ^trig ]]; then
    return 0
  fi
  local fwd_args=()
  [[ "${SUBMIT_ONLY}" == "1" ]] && fwd_args+=(--submit-only)
  echo "On CPU login (${host}); forwarding to trig-login01..."
  exec ssh -o BatchMode=yes -o ConnectTimeout=15 trig-login01 \
    "cd $(printf '%q' "${PROJECT_ROOT}") && STONKBENCH_RUN_ID=$(printf '%q' "${STONKBENCH_RUN_ID}") GENERATION_LENGTH=${GENERATION_LENGTH} NUM_SAMPLES=${NUM_SAMPLES} bash scripts/run_sig_timegan_train.sh $(printf '%q ' "${fwd_args[@]}")"
}

sbatch_gpu() {
  /opt/slurm/bin/sbatch --export=NONE --get-user-env \
    --export=STONKBENCH_SMOKE=0,STONKBENCH_DEVICE=cuda,STONKBENCH_RUN_ID="${STONKBENCH_RUN_ID}",GENERATION_LENGTH="${GENERATION_LENGTH}",NUM_SAMPLES="${NUM_SAMPLES}" \
    "$@"
}

sync_to_home() {
  local run_id="${STONKBENCH_RUN_ID}"
  if [[ ! -d "${STAGING_ROOT}/results/${run_id}" ]]; then
    echo "No staged results at ${STAGING_ROOT}/results/${run_id}"
    exit 1
  fi
  mkdir -p "${CANONICAL_OUTPUT}/results/${run_id}" \
    "${CANONICAL_OUTPUT}/checkpoints/${run_id}" \
    "${CANONICAL_OUTPUT}/logs/${run_id}" \
    "${CANONICAL_OUTPUT}/sanity/${run_id}" \
    "${CANONICAL_OUTPUT}/experiments/${run_id}"

  echo "Syncing run ${run_id} (no --delete of sibling dated runs)"
  rsync -av "${STAGING_ROOT}/results/${run_id}/" "${CANONICAL_OUTPUT}/results/${run_id}/"
  [[ -d "${STAGING_ROOT}/checkpoints/${run_id}" ]] && \
    rsync -av "${STAGING_ROOT}/checkpoints/${run_id}/" "${CANONICAL_OUTPUT}/checkpoints/${run_id}/"
  [[ -d "${STAGING_ROOT}/logs/${run_id}" ]] && \
    rsync -av "${STAGING_ROOT}/logs/${run_id}/" "${CANONICAL_OUTPUT}/logs/${run_id}/"
  [[ -d "${STAGING_ROOT}/sanity/${run_id}" ]] && \
    rsync -av "${STAGING_ROOT}/sanity/${run_id}/" "${CANONICAL_OUTPUT}/sanity/${run_id}/"
  [[ -d "${STAGING_ROOT}/experiments/${run_id}" ]] && \
    rsync -av "${STAGING_ROOT}/experiments/${run_id}/" "${CANONICAL_OUTPUT}/experiments/${run_id}/"

  echo ""
  echo "Artifacts in home for run ${run_id}:"
  find "${CANONICAL_OUTPUT}/results/${run_id}" -path '*/artifacts/*.pt' 2>/dev/null | sort
  local n
  n=$(find "${CANONICAL_OUTPUT}/results/${run_id}" -path '*/artifacts/*.pt' 2>/dev/null | wc -l)
  echo "Total artifacts: ${n}"
}

submit_pipeline() {
  require_gpu_login
  mkdir -p "${SCRATCH_ROOT}/stonkbench/slurm_logs" "${STAGING_ROOT}/results/${STONKBENCH_RUN_ID}"
  cd "${PROJECT_ROOT}"

  local hp_id agg_id dl_id
  hp_id=$(sbatch_gpu --parsable "${SLURM_DIR}/hp_search_sig_timegan.sh")
  agg_id=$(sbatch_gpu --parsable --dependency=afterany:"${hp_id}" "${SLURM_DIR}/hp_aggregate.sh")
  dl_id=$(sbatch_gpu --parsable --dependency=afterok:"${agg_id}" "${SLURM_DIR}/final_training_sig_timegan.sh")

  echo "Submitted SIG+TimeGAN train pipeline (no eval, no stats):"
  echo "  RUN_ID=${STONKBENCH_RUN_ID}"
  echo "  models=timegan pcf_gan sig_wgan"
  echo "  gen_length=${GENERATION_LENGTH} num_samples=${NUM_SAMPLES}"
  echo "  1 HP search (27 x 1 GPU, max 8): ${hp_id}"
  echo "  2 HP aggregate:                  ${agg_id}  (afterany:${hp_id})"
  echo "  3 DL final train (3 models):     ${dl_id}  (afterok:${agg_id})"
  echo ""
  echo "Scratch results: ${STAGING_ROOT}/results/${STONKBENCH_RUN_ID}"
  echo "Home results:    ${CANONICAL_OUTPUT}/results/${STONKBENCH_RUN_ID}"
  echo "Logs: ${SCRATCH_ROOT}/stonkbench/slurm_logs/"
  echo "${hp_id} ${agg_id} ${dl_id}" > "${SCRATCH_ROOT}/stonkbench/sig_timegan_job_ids.txt"
  echo "${STONKBENCH_RUN_ID}" > "${SCRATCH_ROOT}/stonkbench/sig_timegan_run_id.txt"

  nohup bash -c "
    IDS=\$(cat ${SCRATCH_ROOT}/stonkbench/sig_timegan_job_ids.txt)
    RUN=${STONKBENCH_RUN_ID}
    while squeue -u \$USER -h | grep -E \"\$(echo \$IDS | tr ' ' '|')\" >/dev/null 2>&1; do
      date -Is
      squeue -u \$USER -o '%.18i %.12P %.12j %.2t %.10M %R' | head -20
      sleep 300
    done
    cd ${PROJECT_ROOT}
    STONKBENCH_RUN_ID=\$RUN bash scripts/run_sig_timegan_train.sh sync
  " > "${SCRATCH_ROOT}/stonkbench/slurm_logs/watch_and_sync_${STONKBENCH_RUN_ID}.log" 2>&1 &

  if [[ "${SUBMIT_ONLY}" == "1" ]]; then
    echo "Submit-only done. Watcher syncing to home when finished."
    return 0
  fi

  echo "Waiting for HP search (array ${hp_id})..."
  while squeue -j "${hp_id}" -h 2>/dev/null | grep -q .; do sleep 60; done
  echo "Waiting for HP aggregate (job ${agg_id})..."
  while squeue -j "${agg_id}" -h 2>/dev/null | grep -q .; do sleep 60; done
  echo "Waiting for DL final train (array ${dl_id})..."
  while squeue -j "${dl_id}" -h 2>/dev/null | grep -q .; do sleep 60; done
  sync_to_home
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    sync) sync_to_home; exit 0 ;;
    --submit-only) SUBMIT_ONLY=1; shift ;;
    -h|--help)
      echo "Usage: STONKBENCH_RUN_ID=<id> bash scripts/run_sig_timegan_train.sh [--submit-only | sync]"
      exit 0
      ;;
    *)
      echo "Usage: STONKBENCH_RUN_ID=<id> bash scripts/run_sig_timegan_train.sh [--submit-only | sync]"
      exit 1
      ;;
  esac
done

submit_pipeline
