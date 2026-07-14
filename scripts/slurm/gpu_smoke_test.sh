#!/bin/bash
#SBATCH --job-name=stonkbench-smoke
#SBATCH --account=def-yqhuang
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/stonkbench/slurm_logs/gpu_smoke_%j.out
#SBATCH --error=/scratch/%u/stonkbench/slurm_logs/gpu_smoke_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/slurm/common.sh"

cd "${PROJECT_ROOT}"
SMOKE_OUT="${OUTPUT_ROOT}/smoke_${SLURM_JOB_ID}"
mkdir -p "${SMOKE_OUT}"

echo "=== StonkBench device smoke test ==="
echo "Host: $(hostname)"
echo "Date: $(date -Is)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found on this node"
fi

python - <<'PY'
import torch
from src.utils.device import get_device, log_device_context

print(log_device_context())
print(f"torch={torch.__version__} cuda_built={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
device = get_device()
x = torch.randn(4, 4, device=device)
y = x @ x.T
print(f"matmul on {device}: shape={tuple(y.shape)} finite={bool(torch.isfinite(y).all())}")
PY

python -m src.experiments.run_benchmark \
  --smoke_test \
  --models quantgan gbm_adapter \
  --generation_length 100 \
  --num_samples 8 \
  --num_epochs 1 \
  --device cuda \
  --output_root "${SMOKE_OUT}"

echo "=== Smoke test complete ==="
