# StonkBench: Unified Benchmark for Synthetic Data Generation in Financial Time Series (SDGFTS)

> A unified, reproducible benchmark for evaluating synthetic time series generators in finance.

---

## Quickstart (CCDB / SciNet Trillium)

### 1. Create the `stonkbench` conda environment

Do **not** use the legacy `stonk` environment.

```bash
source ~/miniconda/bin/activate
cd ~/StonkBench
bash scripts/install_stonkbench.sh
export PYTHONPATH=.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

On **GPU compute nodes**, reinstall PyTorch with CUDA 12.6 wheels after activation:

```bash
pip install --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

Device resolution is automatic via `src/utils/device.py` (`cuda` when available, else `cpu`).

### 2. Download and preprocess data

```bash
python src/data_downloader.py --index SPY QQQ IWM XLF XLV AAPL MSFT NVDA AVGO JPM LLY UNH AMZN TSLA CAT UNP META NFLX PG COST XOM CVX NEE PLD LIN --start 2023-01-01 --end 2025-01-01

python src/data_preprocessing.py \
  --input_csv data/combined_data.csv \
  --output_dir data/preprocessed \
  --window_size 21 \
  --stride 1 \
  --train_ratio 0.8
```

Outputs:

- `data/preprocessed/dl_set.pt` — sliding windows `(N, L, C)` for deep learning adapters
- `data/preprocessed/statsmodel_set.pt` — full train series `(T, C)` for statistical adapters

### 3. Run locally (CPU or GPU)

```bash
python -m src.experiments.run_benchmark \
  --models quantgan gbm_adapter \
  --generation_length 52 \
  --num_samples 128 \
  --num_epochs 3 \
  --smoke_test
```

Device is resolved automatically (`cuda` when available, else `cpu`). Override with `--device cpu` or `export STONKBENCH_DEVICE=cuda`.

### 4. Submit to Slurm (Neptune nodes)

CCDB Trillium has **no GPU GRES** in Slurm (`sinfo` shows `GRES=(null)` on all partitions). The largest CPU nodes are **Neptune** (`compute_neptune`, 160 logical cores, ~467 GiB) and **tri** (`compute`, 192 cores, ~745 GiB). Neptune jobs require `--qos=neptune`.

Slurm scripts write logs to `$SCRATCH/stonkbench/logs/` (home is read-only on compute nodes). Do not pass `--mem` on Trillium. Jobs request a full node (`--nodes=1`).

If `sbatch` fails with *"User is not known to the scheduler"*, CCDB may not have synced yet (up to ~24h after being added to `def-yqhuang`). Verify with:

```bash
sacctmgr show assoc user=$USER format=Account,QOS,Partition
```

| Script | Purpose |
|--------|---------|
| `run_benchmark.sh` | Single generation-length benchmark |
| `run_benchmark_array.sh` | Array over seq lengths 52–300 (replaces `parallelizer_script.py`) |
| `run_eval.sh` | Unified evaluation |
| `gpu_smoke_test.sh` | Minimal training + device probe |

```bash
mkdir -p $SCRATCH/stonkbench/logs
sbatch scripts/slurm/gpu_smoke_test.sh          # quick device + training check
sbatch scripts/slurm/run_benchmark.sh
sbatch scripts/slurm/run_benchmark_array.sh
sbatch --dependency=afterok:<array_job_id> scripts/slurm/run_eval.sh
```

All Slurm scripts use `--account=def-yqhuang` by default. Override with `sbatch --account=...` if needed.

Hyperparameter search array:

```bash
sbatch scripts/hp_search_slurm.sh
```

---

## Architecture

### Data flow

`data_downloader.py` → `data_preprocessing.py` → adapter training/generation → `unified_evaluator.py`

### Statistical vs deep learning inputs

| Kind | Preprocessed file | Tensor shape | Used by |
|------|-------------------|--------------|---------|
| Statistical | `statsmodel_set.pt` | `(T, C)` train series | GBM, GARCH, bootstrap, … |
| Deep learning | `dl_set.pt` | `(N, L, C)` windows | QuantGAN, TimeGAN, TimeVAE, … |

`data_preprocessing.py` writes both artifacts from one CSV pass; adapters select the correct format via `STATISTICAL_MODEL_KEYS` in the pipeline.

### Device strategy

One module — `src/utils/device.py` — provides `get_device()` / `resolve_device()`:

- Entry points (`run_benchmark`, `hp_search`, pipeline) resolve the device once and pass it through `AdapterFitInput.device`.
- DL adapters call `resolve_device()` before `.to(device)` and DataLoader transfers.
- No scattered `torch.cuda.is_available()` branches elsewhere.

### Adapter layer

External repos stay vendored under `src/models/deep_learning/`; adapters in `src/experiments/adapters/` wrap clone-and-train APIs without removing the shim pattern.

---

## Evaluate and plot

```bash
python src/unified_evaluator.py \
  --generated_dir src/experiments \
  --results_dir results \
  --seq_lengths 52 60 120 180 240 300

python src/plot_statistics/evaluation_plotter.py
```

Smoke test:

```bash
python -m src.experiments.smoke_test
```

---

## Docker (optional)

```bash
docker-compose build base
docker-compose up
```

---

## Project structure

```
StonkBench/
  data/                          # Raw + preprocessed datasets
  results/                       # Evaluation JSON
  evaluation_plots/              # Figures
  scripts/slurm/                 # Slurm submission scripts
  src/
    utils/device.py              # Unified CPU/GPU resolution
    data_downloader.py
    data_preprocessing.py
    experiments/
      run_benchmark.py           # Main training/generation entry
      adapters/                  # Statistical + DL adapter shims
    unified_evaluator.py
    models/
  requirements.txt
```

---

## Contributors

| Name | Role | Email |
|------|------|-------|
| **Eddison Pham** | ML Researcher & Engineer | eddison.pham@mail.utoronto.ca |
| **Albert Lam Ho** | Quantitative Researcher | uyenlam.ho@mail.utoronto.ca |
| **Yiqing Irene Huang** | Research Supervisor | iy.huang@mail.utoronto.ca |
