#!/usr/bin/env bash
# Create the stonkbench conda environment on CCDB (SciNet Trillium GPU login).
# Run from trig-login01 (home is writable there).
set -euo pipefail

source "${HOME}/miniconda/bin/activate"
conda create -n stonkbench python=3.11 -y
conda activate stonkbench

# Scientific stack from conda-forge (avoids Compute Canada pip wheel issues).
conda install -y -c conda-forge \
  numpy=2.2.2 scipy pandas scikit-learn matplotlib seaborn statsmodels arch-py \
  joblib tqdm pyyaml xgboost requests toolz typing_extensions pydantic \
  lightning pytorch-lightning einops opt_einsum level-zero

# Official torch+cu126 wheel. Alliance pip prefers CC wheelhouse, so download
# the manylinux wheel and rename to linux_x86_64 before installing.
WHEEL_DIR="${SCRATCH:-/scratch/$USER}/stonkbench/wheels"
mkdir -p "${WHEEL_DIR}"
curl -L --fail --retry 3 -A "Mozilla/5.0" \
  -o "${WHEEL_DIR}/torch-2.6.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl" \
  "https://download.pytorch.org/whl/cu126/torch-2.6.0%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl"
cp -f "${WHEEL_DIR}/torch-2.6.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl" \
  "${WHEEL_DIR}/torch-2.6.0+cu126-cp311-cp311-linux_x86_64.whl"
PIP_CONFIG_FILE=/dev/null pip install --force-reinstall --no-deps \
  "${WHEEL_DIR}/torch-2.6.0+cu126-cp311-cp311-linux_x86_64.whl"

# Remaining pip-only packages.
pip install --no-deps gluonts==0.16.2 dtaidistance==2.3.9
pip install yfinance==1.5.1

echo "stonkbench ready. Activate with: conda activate stonkbench"
echo "Jobs load CUDA via scripts/slurm/common.sh (module load cuda/12.6 + LD_LIBRARY_PATH)."
