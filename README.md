# StonkBench — unified benchmark for synthetic financial time series

A reproducible benchmark for **generating** and **evaluating** synthetic
multivariate financial time series: 12 generator models, 25 assets, and four
metric families (fidelity, diversity, stylised facts, P&L) plus two downstream
tasks (portfolio construction, option hedging).

```bash
bash scripts/check_repo.sh        # preflight: syntax, unit tests, docs, bundle
```

---

## Quickstart

### 1. Environment (once)

```bash
bash scripts/install_stonkbench.sh
source ~/.venvs/stonkbench/bin/activate
export PYTHONPATH=.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # only inside a conda shell
```

`scripts/install_stonkbench.sh` builds `~/.venvs/stonkbench` with the scientific
stack and a CUDA torch wheel. `scripts/slurm/common.sh` activates that same venv,
so Slurm jobs and local runs share one environment. To use PyTorch on CPU only:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Device selection is automatic (`src/utils/device.py` picks `cuda` when available,
else `cpu`); override with `--device cpu` or `export STONKBENCH_DEVICE=cuda`.

### 2. Data (once)

```bash
python src/data_downloader.py \
  --index SPY QQQ IWM XLF XLV AAPL MSFT NVDA AVGO JPM LLY UNH AMZN TSLA CAT \
         UNP META NFLX PG COST XOM CVX NEE PLD LIN \
  --start 2010-01-01 --end 2025-01-01

python src/data_preprocessing.py \
  --input_csv data/combined_data.csv \
  --output_dir data/preprocessed \
  --window_size 100 --stride 1 --train_ratio 0.8
```

This writes both formats the adapters need:

| file | shape | used by |
|------|-------|---------|
| `data/preprocessed/dl_set.pt` | `(N, L, C)` windows | deep learning models |
| `data/preprocessed/statsmodel_set.pt` | `(T, C)` series | statistical models |

### 3. Train, generate, evaluate

```bash
# quick sanity run (CPU or GPU, a few minutes)
python -m src.experiments.run_benchmark \
  --models quantgan gbm_adapter --generation_length 100 \
  --num_samples 128 --num_epochs 3 --smoke_test

# evaluate the generated artifacts
python -m src.unified_evaluator \
  --generated_dir outputs/results/latest \
  --results_dir outputs/results/latest/evaluation \
  --seq_lengths 100 --skip_regenerate
```

Everything a run produces lands under `outputs/`; re-running with the same
`STONKBENCH_RUN_ID` (default `latest`) overwrites in place.

---

## Pipeline at a glance

| stage | command | notes |
|-------|---------|-------|
| preflight | `bash scripts/check_repo.sh` | run before and after any rerun |
| data | `python src/data_downloader.py` → `python src/data_preprocessing.py` | once per date range |
| hyperparameter search | `python -m src.experiments.hp_search --models <keys> --smoke` | `--list_trials` shows the grid |
| train + generate | `python -m src.experiments.run_benchmark` | the main entry point |
| final training | `python -m src.experiments.run_final_training --hp_summary <csv>` | from a search summary |
| evaluate | `python -m src.unified_evaluator` | all four metric families + downstream |
| figures | `python src/plot_statistics/evaluation_plotter.py` | writes `evaluation_plots/` |
| smoke test | `bash scripts/slurm/gpu_smoke_test.sh` | minimal training + device probe |
| end-to-end | `bash scripts/run_pipeline.sh --smoke --submit-only` | search → train → eval |

Run any entry point with `--help` (or `-h`) for the full flag list. Every stage
honours `PYTHONPATH=.` and resolves paths through `src/utils/env.py`, which
accepts `STONKBENCH_OUTPUT_ROOT`, `STONKBENCH_DL_SET_PATH`,
`STONKBENCH_STATS_SET_PATH` and `STONKBENCH_RUN_ID`.

---

## Slurm (CCDB / SciNet Trillium)

CCDB Trillium has **no GPU GRES** (`sinfo` shows `GRES=(null)` on every
partition). The largest CPU nodes are **Neptune** (`compute_neptune`, 160 logical
cores) and **tri** (`compute`, 192 cores); Neptune jobs need `--qos=neptune`.
Jobs request a full node and never pass `--mem`. Logs go to `slurm_logs/` and
artifacts to `outputs/` — there is no scratch staging tree.

```bash
mkdir -p slurm_logs
sbatch scripts/slurm/gpu_smoke_test.sh        # device + training check first
sbatch scripts/slurm/run_benchmark.sh         # one generation length
sbatch scripts/slurm/run_benchmark_array.sh   # array over sequence lengths
sbatch --dependency=afterok:<array_id> scripts/slurm/eval.sh
```

| script | purpose |
|--------|---------|
| `scripts/slurm/gpu_smoke_test.sh` | minimal training + device probe |
| `scripts/slurm/run_benchmark.sh` | single benchmark run |
| `scripts/slurm/run_benchmark_array.sh` | array over sequence lengths |
| `scripts/slurm/eval.sh` | unified evaluation |
| `scripts/slurm/eval_all.sh` / `eval_parallel.sh` / `eval_full.sh` | full / parallel evaluation |
| `scripts/slurm/hp_search.sh` | hyperparameter search array |
| `scripts/slurm/hp_aggregate.sh` | aggregate search results |
| `scripts/slurm/final_training*.sh` | final training (all / DL / statistical) |
| `scripts/slurm/run_all_models.sh` | every model in one submission |

All scripts default to `--account=def-yqhuang`; override with
`sbatch --account=...`. If `sbatch` reports *"User is not known to the
scheduler"*, CCDB may need up to 24h to sync a new account:

```bash
sacctmgr show assoc user=$USER format=Account,QOS,Partition
```

---

## Docker (optional)

```bash
docker compose build base     # image with the full dependency stack
docker compose up data-download data-preprocess   # data pipeline
docker compose up generate-data eval plot          # generate -> evaluate -> figures
```

`docker-compose.yml` wires the stages together with healthchecks and shared
volumes; each service is `docker compose run <service>` for a one-off run. GPU
passthrough needs `CUDA_VISIBLE_DEVICES` set on the host.

---

## Tests and preflight

```bash
bash scripts/check_repo.sh                       # everything below in one command
python3 -m unittest discover -s tests -v         # unit tests (numpy only, no torch)
pytest tests/                                    # same suite, if you prefer pytest
```

The suite covers the P&L and portfolio evaluators — compounding conventions,
undefined-ratio handling, the optimiser stalling at 1/N, real-baseline key
collisions, turnover — and the bundle reader in `scripts/handover/`.
`tests/_bootstrap.py` installs a torch stub so nothing here needs a GPU or a
PyTorch install.

---

## Evaluation results and the handover bundle

`handover/` is a data-only bundle with the trained checkpoints, the generated
samples, the real data each cell was scored against, and the full evaluation JSON
for 48 cells (12 models × 4 sequence lengths). Its docs are the fastest way into
the numbers:

* [`handover/README.md`](handover/README.md) — what is in it
* [`handover/GUIDE.md`](handover/GUIDE.md) — recipes for comparisons, ablations
  and visualisations, including a task table for coding agents
* [`handover/KNOWN_ISSUES.md`](handover/KNOWN_ISSUES.md) — which evaluator
  defects were found and fixed, and what is still unfixed

Regenerate and verify it with:

```bash
python3 scripts/handover/recompute_downstream.py   # corrected P&L + portfolio blocks
python3 scripts/handover/build_manifest.py
python3 scripts/handover/verify_bundle.py
```

---

## Project structure

```
StonkBench/
  data/                        raw + preprocessed datasets (gitignored)
  outputs/                     every run's artifacts (gitignored)
  handover/                    evaluation bundle + its docs
  scripts/
    check_repo.sh              preflight for any rerun
    install_stonkbench.sh      build the ~/.venvs/stonkbench environment
    run_pipeline.sh            end-to-end submission driver
    slurm/                     Slurm submission scripts
    handover/                  bundle tooling (reader, verifier, recomputation)
  src/
    data_downloader.py         fetch price history
    data_preprocessing.py      one CSV pass -> dl_set.pt + statsmodel_set.pt
    experiments/               run_benchmark, hp_search, run_final_training, smoke_test
      adapters/                statistical + DL adapter shims
    models/                    statistical models and vendored DL repos
    taxonomies/                fidelity, diversity, stylised facts, P&L, portfolio
    unified_evaluator.py       the evaluation entry point
    plot_statistics/           figures
    utils/                     device, env paths, preprocessing helpers
  tests/                       unit tests (torch-free)
  requirements.txt
```

External model repos stay vendored under `src/models/deep_learning/`; adapters in
`src/experiments/adapters/` wrap their clone-and-train APIs. One module,
`src/utils/device.py`, resolves the device and passes it down — there are no
scattered `torch.cuda.is_available()` branches.

---

## Contributors

| Name | Role | Email |
|------|------|-------|
| **Eddison Pham** | ML Researcher & Engineer | eddison.pham@mail.utoronto.ca |
| **Albert Lam Ho** | Quantitative Researcher | uyenlam.ho@mail.utoronto.ca |
| **Yiqing Irene Huang** | Research Supervisor | iy.huang@mail.utoronto.ca |
