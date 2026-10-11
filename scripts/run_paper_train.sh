#!/bin/bash
# Paper training pipeline (NO evaluation):
#   1) DL HP sweep (72 trials = 8 models x 9 configs)
#   2) HP aggregate
#   3) DL final train (gen_length=100) on best HP
#   4) Statistical fit+generate (gen_length=100)
#   5) Report output status for the dated run
#
# Usage:
#   STONKBENCH_RUN_ID=2026-07-11_window100 bash scripts/run_paper_train.sh --submit-only
#   bash scripts/run_paper_train.sh status
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
# Single canonical output tree: <repo>/outputs/. There is no scratch staging
# copy and no post-hoc rsync — training jobs write here directly.
CANONICAL_OUTPUT="${STONKBENCH_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
SLURM_DIR="${PROJECT_ROOT}/scripts/slurm"
SUBMIT_ONLY=0
GENERATION_LENGTH="${GENERATION_LENGTH:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
# Run-id token. Default "latest" so re-runs overwrite in-place; pass
# STONKBENCH_RUN_ID=<custom> for an isolated scope.
STONKBENCH_RUN_ID="${STONKBENCH_RUN_ID:-latest}"
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
    "cd $(printf '%q' "${PROJECT_ROOT}") && STONKBENCH_RUN_ID=$(printf '%q' "${STONKBENCH_RUN_ID}") GENERATION_LENGTH=${GENERATION_LENGTH} NUM_SAMPLES=${NUM_SAMPLES} bash scripts/run_paper_train.sh $(printf '%q ' "${fwd_args[@]}")"
}

sbatch_gpu() {
  /opt/slurm/bin/sbatch --export=NONE --get-user-env \
    --export=STONKBENCH_SMOKE=0,STONKBENCH_DEVICE=cuda,STONKBENCH_RUN_ID="${STONKBENCH_RUN_ID}",GENERATION_LENGTH="${GENERATION_LENGTH}",NUM_SAMPLES="${NUM_SAMPLES}" \
    "$@"
}

report_status() {
  local run_id="${STONKBENCH_RUN_ID}"
  if [[ ! -d "${CANONICAL_OUTPUT}/results/${run_id}" ]]; then
    echo "No results at ${CANONICAL_OUTPUT}/results/${run_id}"
    exit 1
  fi

  echo ""
  echo "Artifacts for run ${run_id}:"
  find "${CANONICAL_OUTPUT}/results/${run_id}" -path '*/artifacts/*.pt' 2>/dev/null | sort
  local n
  n=$(find "${CANONICAL_OUTPUT}/results/${run_id}" -path '*/artifacts/*.pt' 2>/dev/null | wc -l)
  echo "Total artifacts: ${n}"
}

wait_for_array() {
  local job_id="$1"
  local label="$2"
  echo "Waiting for ${label} (array ${job_id})..."
  while squeue -j "${job_id}" -h 2>/dev/null | grep -q .; do
    sleep 60
  done
  echo "${label} finished"
}

wait_for_job() {
  local job_id="$1"
  local label="$2"
  echo "Waiting for ${label} (job ${job_id})..."
  while squeue -j "${job_id}" -h 2>/dev/null | grep -q .; do
    sleep 60
  done
  local state exit_code
  state=$(sacct -j "${job_id}" --format=State --noheader -P 2>/dev/null | head -1 | cut -d'|' -f1 | tr -d ' ')
  exit_code=$(sacct -j "${job_id}" --format=ExitCode --noheader -P 2>/dev/null | head -1 | cut -d'|' -f1 | tr -d ' ')
  echo "${label}: state=${state} exit=${exit_code}"
}

submit_pipeline() {
  require_gpu_login
  mkdir -p "${PROJECT_ROOT}/slurm_logs" "${CANONICAL_OUTPUT}/results/${STONKBENCH_RUN_ID}"
  cd "${PROJECT_ROOT}"

  local hp_id agg_id dl_id stat_id
  hp_id=$(sbatch_gpu --parsable "${SLURM_DIR}/hp_search.sh")
  agg_id=$(sbatch_gpu --parsable --dependency=afterany:"${hp_id}" "${SLURM_DIR}/hp_aggregate.sh")
  dl_id=$(sbatch_gpu --parsable --dependency=afterok:"${agg_id}" "${SLURM_DIR}/final_training_dl.sh")
  stat_id=$(sbatch_gpu --parsable --dependency=afterok:"${agg_id}" "${SLURM_DIR}/final_training_stat.sh")

  echo "Submitted PAPER TRAIN pipeline (no eval):"
  echo "  RUN_ID=${STONKBENCH_RUN_ID}"
  echo "  gen_length=${GENERATION_LENGTH}"
  echo "  1 DL HP search (72 x 1 GPU, max 8): ${hp_id}"
  echo "  2 HP aggregate:                     ${agg_id}"
  echo "  3 DL final train (8 models):        ${dl_id}  (afterok:${agg_id})"
  echo "  4 Statistical fit (6 models):       ${stat_id}  (afterok:${agg_id})"
  echo ""
  echo "Results: ${CANONICAL_OUTPUT}/results/${STONKBENCH_RUN_ID}"
  echo "Logs:    ${PROJECT_ROOT}/slurm_logs/"
  echo "${hp_id} ${agg_id} ${dl_id} ${stat_id}" > "${PROJECT_ROOT}/slurm_logs/paper_train_job_ids.txt"
  echo "${STONKBENCH_RUN_ID}" > "${PROJECT_ROOT}/slurm_logs/paper_train_run_id.txt"

  # Watcher reports status for this run_id when jobs finish.
  nohup bash -c "
    IDS=\$(cat ${PROJECT_ROOT}/slurm_logs/paper_train_job_ids.txt)
    RUN=${STONKBENCH_RUN_ID}
    while squeue -u \$USER -h | grep -E \"\$(echo \$IDS | tr ' ' '|')\" >/dev/null 2>&1; do
      date -Is
      squeue -u \$USER -o '%.18i %.12P %.12j %.2t %.10M %R' | head -20
      sleep 300
    done
    cd ${PROJECT_ROOT}
    STONKBENCH_RUN_ID=\$RUN bash scripts/run_paper_train.sh status
  " > "${PROJECT_ROOT}/slurm_logs/watch_and_report_${STONKBENCH_RUN_ID}.log" 2>&1 &

  if [[ "${SUBMIT_ONLY}" == "1" ]]; then
    echo "Submit-only done. Watcher reports status when finished."
    return 0
  fi

  wait_for_array "${hp_id}" "HP search"
  wait_for_job "${agg_id}" "HP aggregate"
  wait_for_array "${dl_id}" "DL final training"
  wait_for_array "${stat_id}" "Statistical training"
  report_status
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    status) report_status; exit 0 ;;
    --submit-only) SUBMIT_ONLY=1; shift ;;
    -h|--help)
      echo "Usage: STONKBENCH_RUN_ID=<id> bash scripts/run_paper_train.sh [--submit-only | sync]"
      exit 0
      ;;
    *)
      echo "Usage: STONKBENCH_RUN_ID=<id> bash scripts/run_paper_train.sh [--submit-only | sync]"
      exit 1
      ;;
  esac
done

submit_pipeline
