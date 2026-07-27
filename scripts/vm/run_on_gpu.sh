#!/usr/bin/env bash
# scripts/vm/run_on_gpu.sh
#
# GPU dispatcher for VM (non-SLURM) execution. Pins a process to one physical
# GPU by index and clamps host-thread counts so that N parallel jobs on the
# same node don't oversubscribe the CPUs or fight over the same CUDA device.
#
# Usage:
#   bash scripts/vm/run_on_gpu.sh <slot> -- <command...>
#   bash scripts/vm/run_on_gpu.sh 0 -- python -m src.experiments.run_benchmark --models quantgan
#
# Slot math: `slot % N_GPUS` chooses the GPU index to attach to. Override the
# detected count with the `N_GPUS` env var if `nvidia-smi` isn't available
# (e.g. CPU-only smoke runs).
#
# Env vars honored:
#   N_GPUS                       Override GPU count.
#   CUDA_VISIBLE_DEVICES         Override the auto-selected GPU.
#   OMP/MKL/OPENBLAS/TORCH_NUM_THREADS   Thread clamps (default 2).
#   STONKBENCH_DEVICE            Default `cuda`; flip to `cpu` for stats models.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <slot> -- <command...>" >&2
    exit 1
fi

SLOT="${1}"
shift || true

# Allow callers to pass `--` after slot; drop it for cleanliness.
if [[ "${1:-}" == "--" ]]; then
    shift
fi

# Detect GPUs (best-effort). Falls back to 1 if `nvidia-smi` is unavailable.
if [[ -z "${N_GPUS:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    N_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
N_GPUS="${N_GPUS:-1}"

# Validate N_GPUS is a positive integer.
if ! [[ "${N_GPUS}" =~ ^[0-9]+$ ]] || [[ "${N_GPUS}" -lt 1 ]]; then
    N_GPUS=1
fi

GPU_IDX=$(( SLOT % N_GPUS ))
# Caller-supplied CUDA_VISIBLE_DEVICES always wins.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDX}}"

# Throttle per-process thread usage (don't override if the user set them already).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-2}"

# Default device hint; pipeline honors `STONKBENCH_DEVICE` via src/utils/device.py.
export STONKBENCH_DEVICE="${STONKBENCH_DEVICE:-cuda}"

# Operational banner to stderr so per-job logs show the assignment.
{
    echo "[run_on_gpu] slot=${SLOT} n_gpus=${N_GPUS} cuda_visible_devices=${CUDA_VISIBLE_DEVICES} cmd=$*"
} >&2

exec "$@"
