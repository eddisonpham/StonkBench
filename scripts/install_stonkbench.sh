#!/usr/bin/env bash
# Create the stonkbench conda environment on CCDB (SciNet Trillium).
set -euo pipefail

source "${HOME}/miniconda/bin/activate"
conda create -n stonkbench python=3.11 -y
conda activate stonkbench

# Scientific stack from conda-forge (avoids Compute Canada pip wheel libcpupower issues on login nodes).
conda install -y -c conda-forge \
  numpy=2.2.2 scipy pandas scikit-learn matplotlib seaborn statsmodels arch-py \
  joblib tqdm pyyaml xgboost requests toolz typing_extensions pydantic \
  lightning pytorch-lightning einops opt_einsum

# PyTorch: CPU wheel for login-node dev; on GPU compute nodes reinstall cu126 if needed:
#   pip install --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Remaining pip-only packages. Keep GluonTS dependency resolution explicit so
# pip does not pull incompatible Compute Canada wheels for core numerics.
pip install --no-deps gluonts==0.16.2 dtaidistance==2.3.9
pip install yfinance==1.5.1

echo "stonkbench ready. Activate with: conda activate stonkbench"
echo "On CCDB login nodes, also: export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
