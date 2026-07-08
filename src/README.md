# `src/`

Core pipeline code for StonkBench.

| Path | Role |
|------|------|
| `data_downloader.py` | Fetch Yahoo Finance CSV |
| `data_preprocessing.py` | Build `dl_set.pt` + `statsmodel_set.pt` |
| `experiments/run_benchmark.py` | Main training/generation entry |
| `experiments/adapters/` | Statistical + DL model shims |
| `experiments/core/pipeline.py` | Orchestration + device routing |
| `utils/device.py` | Unified CPU/GPU resolution |
| `unified_evaluator.py` | Metric evaluation |

Slurm scripts: `../scripts/slurm/`. Environment: `stonkbench` conda env (`../scripts/install_stonkbench.sh`).
