#!/bin/bash
# Local login-node evaluation runner.
#
# Runs the full 48-task grid (12 models x 4 seq lengths) DIRECTLY on the login
# node — bypassing the congested scheduler — using the node's local GPUs
# (round-robin across them) and its many cores. Each task evaluates exactly ONE
# artifact (native seq length) thanks to the exact-match seq filter in
# src/unified_evaluator.py, so there are no trim races on metrics.json.
#
# Usage:
#   bash scripts/local_eval_runner.sh
#   LOCAL_EVAL_CONCURRENCY=16 bash scripts/local_eval_runner.sh   # more parallel
#   LOCAL_EVAL_THREADS=4  bash scripts/local_eval_runner.sh       # fewer threads/task
set -uo pipefail

# Raise the CPU-time soft limit to the hard limit (3600s default soft kills
# heavy stat-model hedging tasks). Non-root users can only reach the hard cap.
ulimit -t "${LOCAL_EVAL_CPU_LIMIT:-5400}" 2>/dev/null || true

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
LOG_DIR="${SCRATCH:-/scratch/$USER}/stonkbench/local_eval"
mkdir -p "${LOG_DIR}"  # must exist before any redirect into it
cd "${PROJECT_ROOT}"
# shellcheck source=scripts/slurm/common.sh
source scripts/slurm/common.sh  # venv, modules, env vars
# common.sh enables ``set -e`` for slurm scripts; neutralize it here so a
# single failing task doesn't kill the whole pool.
set +e

RUN_ID="${STONKBENCH_RUN_ID:-latest}"
SCRATCH_OUT="/scratch/${USER}/stonkbench/output"
if [[ -d "${SCRATCH_OUT}/results/${RUN_ID}" ]]; then
  GENERATED_DIR="${SCRATCH_OUT}/results/${RUN_ID}"
else
  GENERATED_DIR="${OUTPUT_ROOT}/results/${RUN_ID}"
fi
RESULTS_DIR="${GENERATED_DIR}/evaluation"
mkdir -p "${RESULTS_DIR}"

# Artifact dir names on disk (the csigwgan variant lives under cond_sig_wgan;
# its metadata model_name is csigwgan_noise10, which the evaluator reports in
# outputs but which would NOT match the parent-dir artifact filter).
MODELS=(
  quantgan
  timegrad
  kalman_vae_safe
  unconditional_tsdiffusion
  conditional_tsdiffusion
  vrnn_klsched
  cond_sig_wgan
  block_bootstrap
  stationary_block_bootstrap
  merton_jump_diffusion
  de_jump_diffusion
  garch11
)
SEQ_LENGTHS=( 21 42 126 252 )
CONCURRENCY="${LOCAL_EVAL_CONCURRENCY:-12}"
THREADS="${LOCAL_EVAL_THREADS:-8}"
# Per-channel parallel evaluation workers (spawned processes). Speeds up the
# DTW + hedging phases ~10x and keeps per-process CPU time under the login-node
# hard cap, which previously SIGKILL'd long seq_126 stat-model tasks.
EVAL_WORKERS="${LOCAL_EVAL_WORKERS:-8}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export GENERATED_DIR RESULTS_DIR LOG_DIR STONKBENCH_EVAL_WORKERS="${EVAL_WORKERS}"
export EVAL_FAILURES_FILE="${LOG_DIR}/failures.log"
: > "${EVAL_FAILURES_FILE}"

echo "$(date -Is) START runner concurrency=${CONCURRENCY} threads=${THREADS} gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)" \
  >> "${LOG_DIR}/progress.log"

# Skip tasks whose metrics.json already exists and is complete (all six
# top-level categories). Output dir uses the artifact's metadata model_name
# (csigwgan_noise10 for the cond_sig_wgan artifacts).
metric_done() {
  local model="$1"
  local seq="$2"
  local out_model="$model"
  [[ "$model" == "cond_sig_wgan" ]] && out_model="csigwgan_noise10"
  local f="${RESULTS_DIR}/seq_${seq}/${out_model}/metrics.json"
  [[ -f "$f" ]] && grep -q '"FidelityEvaluator"' "$f" 2>/dev/null && grep -q '"utility"' "$f" 2>/dev/null
}

run_one() {
  local model="$1" seq="$2" gpu="$3"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  nice -n 10 python -m src.unified_evaluator \
    --generated_dir "${GENERATED_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --model "${model}" \
    --seq_length "${seq}" \
    --skip_regenerate \
    > "${LOG_DIR}/${model}_seq${seq}.log" 2>&1
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "${model} seq${seq} rc=${rc}" >> "${EVAL_FAILURES_FILE}"
  fi
  echo "$(date -Is) DONE ${model} seq${seq} gpu=${gpu} rc=${rc}" >> "${LOG_DIR}/progress.log"
}
export -f run_one

gpu_counter=0
for m in "${MODELS[@]}"; do
  for s in "${SEQ_LENGTHS[@]}"; do
    if metric_done "${m}" "${s}"; then
      echo "$(date -Is) SKIP ${m} seq${s} (metrics.json exists)" >> "${LOG_DIR}/progress.log"
      continue
    fi
    while (( $(jobs -pr 2>/dev/null | wc -l) >= CONCURRENCY )); do sleep 10; done
    run_one "${m}" "${s}" "$(( gpu_counter % 4 ))" &
    gpu_counter=$(( gpu_counter + 1 ))
  done
done
wait
if [[ -s "${EVAL_FAILURES_FILE}" ]]; then
  echo "$(date -Is) FAILURES: $(tr '\n' ' ' < "${EVAL_FAILURES_FILE}")" >> "${LOG_DIR}/progress.log"
else
  echo "$(date -Is) ALL TASKS DONE, ZERO FAILURES" >> "${LOG_DIR}/progress.log"
fi

# --- Repair utility summaries (recursive nested aggregation) ---
python scripts/repair_utility_summaries.py --results_dir "${RESULTS_DIR}" \
  >> "${LOG_DIR}/progress.log" 2>&1

# --- Merge per-task metrics.json -> complete_evaluation.json ---
python -c "
import json
from pathlib import Path

results_dir = Path('${RESULTS_DIR}')
all_results = {}
for metrics_file in sorted(results_dir.glob('seq_*/*/metrics.json')):
    try:
        with open(metrics_file) as f:
            data = json.load(f)
    except Exception:
        continue
    model = data.get('model_name', '')
    seq = data.get('evaluated_at_length', data.get('sequence_length', 0))
    all_results[f'{model}_seq{seq}'] = data

summary_path = results_dir / 'complete_evaluation.json'
with open(summary_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f'Merged {len(all_results)} results -> {summary_path}')
" >> "${LOG_DIR}/progress.log" 2>&1

# --- Generate plots ---
PLOTS_DIR="${GENERATED_DIR}/plots"
mkdir -p "${PLOTS_DIR}"
python -m src.plot_statistics.evaluation_plots \
  --results_dir "${RESULTS_DIR}" \
  --output_dir "${PLOTS_DIR}" \
  >> "${LOG_DIR}/progress.log" 2>&1

echo "$(date -Is) ALL DONE (merge + plots)" >> "${LOG_DIR}/progress.log"
