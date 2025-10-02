# Unified Benchmark for Synthetic Data Generation in Financial Time Series (SDGFTS)

> A unified, reproducible benchmark to evaluate synthetic time series generators for downstream financial forecasting tasks.

---

## TL;DR

TBD

---

## Abstract

TBD

---

## Key Features

TBD

---

## Repository Structure

```
Unified-benchmark-for-SDGFTS-main/
  ├─ Datasets/                 # Real datasets (or scripts to fetch)
  ├─ dataset_downloader.py     # Helper to download/prep datasets
  ├─ requirements.txt          # Python dependencies
  └─ README.md                 # This file
```

Planned/typical additions:

```
  ├─ src/
  │   ├─ generators/           # Synthetic data generators (interfaces + impls)
  │   ├─ models/               # Forecasting models (classical + deep)
  │   ├─ metrics/              # Fidelity, privacy, utility metrics
  │   ├─ evaluation/           # Protocols, loops, and aggregations
  │   ├─ data/                 # Dataset loaders, preprocessing, scalers
  │   └─ utils/                # Common helpers, logging, io
  ├─ configs/                  # YAML/JSON experiment configs
  ├─ scripts/                  # CLI entrypoints for runs
  └─ results/                  # Run artifacts (optional, gitignored)
```

---

## Datasets

This repo supports three datasets, saved under `Datasets/` in their own folders:

- **GOOG Recent Price History (Daily, last 5 years)** — source: `https://finance.yahoo.com/quote/GOOG/history?p=GOOG` → saved to `Datasets/GOOG_recent/`
  - Fetch: `python dataset_downloader.py --goog-recent`

- **GOOG Long Price History (Daily, max range)** — source: `https://finance.yahoo.com/quote/GOOG/history?p=GOOG` → saved to `Datasets/GOOG_long/`
  - Fetch: `python dataset_downloader.py --goog-long`

- **Multivariate Time Series Datasets (Electricity, Traffic, Solar Energy, Exchange Rate)** — source: `https://github.com/laiguokun/multivariate-time-series-data` → saved to `Datasets/multivariate-time-series-data/`
  - Fetch: `python dataset_downloader.py --mts-repo`

Fetch all at once:

```bash
python dataset_downloader.py --all
```

Each dataset should provide:

- Schema: `timestamp`, `symbol` (optional), feature columns, target column(s)
- Train/val/test splits by time (no leakage)
- Scaling/normalization policy documented

---

## Tasks & Forecasting Horizons

TODO: Describe downstream tasks 

---

## Forecasting Models

TODO: List out the classification of models used.

---

## Evaluation Protocol

TODO: Specify the step-by-step evaluation pipeline.
---

## Metrics

TODO: Define metric suite and reporting format.

---

## Installation

TODO: Specify Python version and installation steps.

```bash
# TODO: set python version and env instructions
pip install -r requirements.txt
```

---

## Quickstart

TODO: Provide an end-to-end minimal example once scripts are ready.

```bash
# TODO: dataset download
# TODO: run benchmark with example config
```

TODO: Describe expected outputs (paths, JSON/CSV schemas).

---

## Reproducing Reported Results

TODO: Link configs and provide exact commands to reproduce tables/figures.

```bash
# TODO: exact commands for main experiments
```

---

## Configuration

TODO: Define config schema (YAML/JSON) and provide examples.

---

## Results & Leaderboard

TODO: Describe output artifacts and leaderboard submission process.

---

## Citation

TODO: Provide citation entry when the paper/preprint is available.

---

## Contributing

TODO: Add contribution guidelines and coding standards.

---

## License

TODO: Specify project license and dataset licensing policy.

---

## Maintainers & Contact

Eddison and Albert

TODO: fill in information

---

