#!/usr/bin/env bash
# Create the stonkbench virtualenv on CCDB (SciNet Trillium GPU login).
# Run from trig-login01 (home is writable there).
#
# scripts/slurm/common.sh activates ${HOME}/.venvs/stonkbench/bin/activate and
# fails if it is absent, so this installer must build that venv. The previous
# conda `-n stonkbench` environment is no longer used by anything.
set -euo pipefail

VENV="${HOME}/.venvs/stonkbench"
python3 -m venv "${VENV}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
python -m pip install --upgrade pip

# Scientific stack (pip-sourced; the conda-forge channel is no longer used).
pip install \
  numpy==2.2.2 scipy pandas scikit-learn matplotlib seaborn statsmodels arch \
  joblib tqdm pyyaml xgboost requests toolz typing_extensions pydantic \
  lightning pytorch-lightning einops opt_einsum

# Official torch+cu126 wheel. Alliance pip prefers CC wheelhouse, so download
# the manylinux wheel and rename to linux_x86_64 before installing.
# Wheel downloads stay inside the repo (previously $SCRATCH/stonkbench/wheels).
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/StonkBench}"
WHEEL_DIR="${PROJECT_ROOT}/.cache/wheels"
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

echo "stonkbench ready. Activate with: source ${VENV}/bin/activate"
echo "Jobs load CUDA via scripts/slurm/common.sh (module load cuda/12.6 + LD_LIBRARY_PATH)."
