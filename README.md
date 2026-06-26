# StonkBench: Unified Benchmark for Synthetic Data Generation in Financial Time Series (SDGFTS)

> A unified, reproducible benchmark for evaluating synthetic time series generators in finance. All results, metrics, and experiment outputs are automatically saved and organized.

---

## Quickstart

### Installation

- Python: 3.11+ (recommended)
- Install all dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Run with Docker Compose (Recommended)

The easiest way to run the complete pipeline is using Docker Compose, which orchestrates all stages from data download to evaluation and plotting.

#### Build the base image

```bash
docker-compose build base
```

#### Run the entire pipeline

```bash
docker-compose up
```

This command runs all services in dependency order:
1. **data-download**: Downloads daily close/volume market data
2. **data-preprocess**: Builds `dl_set.pt` and `statsmodel_set.pt`
3. **generate-data**: Generates synthetic data using both statistical and deep learning models
4. **eval**: Evaluates all generated data using the unified evaluator
5. **plot**: Generates publication-ready figures from evaluation results

#### Run specific services

```bash
# Run only data download
docker-compose up data-download

# Run data download + preprocessing + generation
docker-compose up data-download data-preprocess generate-data

# Run through evaluation, skip plotting
docker-compose up data-download data-preprocess generate-data eval
```

#### Environment variables

Set environment variables via `.env` file or export in your shell:

```bash
# Set the CUDA device (if using CUDA)
export CUDA_VISIBLE_DEVICES=0
```

#### Volume mounts

The following local directories are mapped into containers:

- `./data` → `/data` (downloaded datasets)
- `./generated_data` → `/generated_data` (synthetic data outputs)
- `./results` → `/results` (evaluation results)
- `./evaluation_plots` → `/evaluation_plots` (plots and figures)
- `./configs` → `/app/configs` (read-only configuration files)

### Run Locally (Non-Docker)

#### 1. Download Dataset

Fetch the required dataset:
```bash
python src/data_downloader.py --index SPY QQQ IWM XLF XLV AAPL MSFT NVDA AVGO JPM LLY UNH AMZN TSLA CAT UNP META NFLX PG COST XOM CVX NEE PLD LIN --start 2023-01-01 --end 2025-01-01
```

This saves data to `data/combined_data.csv`.

#### 2. Preprocess Dataset

```bash
python src/data_preprocessing.py \
  --input_csv data/combined_data.csv \
  --output_dir data/preprocessed \
  --window_size 21 \
  --stride 1 \
  --train_ratio 0.8
```

This writes:
- `data/preprocessed/dl_set.pt`
- `data/preprocessed/statsmodel_set.pt`

#### 3. Generate Synthetic Data

Generate synthetic data using the multivariate adapter pipeline:

```bash
python src/generation_scripts/generate_data.py \
  --models quantgan timegan timegrad timevae unconditional_tsdiffusion vrnn gbm_adapter \
  --generation_length 52 \
  --num_samples 1000 \
  --seed 42 \
  --output_root src/experiments
```

The script loads the preprocessed `.pt` datasets, trains through model adapters, and writes artifacts/checkpoints/logs under `src/experiments/<model_name>/`.

#### 4. Evaluate Generated Data

Evaluate all generated artifacts:

```bash
python src/unified_evaluator.py \
  --generated_dir src/experiments \
  --results_dir results \
  --seq_lengths 52 60 120 180 240 300
```

#### 4.1 Smoke Test All Models

```bash
python src/experiments/smoke_test.py
```

Outputs are saved to:
- `/results/seq_<L>/<ModelName>/metrics.json` - Evaluation metrics
- `/results/seq_<L>/<ModelName>/visualizations/` - Visualization outputs

#### 5. Generate Publication-Ready Plots

Generate comprehensive, publication-ready plots for all evaluation metrics:

```bash
python src/plot_statistics/evaluation_plotter.py
```

This automatically finds the latest evaluation results, generates publication-quality plots (300 DPI), and saves them to `evaluation_plots/` directory.

### Pipeline Overview

**What happens:**
- `src/data_preprocessing.py` transforms data:
  - Price columns: log returns
  - Volume columns: log(volume)
- Deep learning models train on precomputed windows `(R, 21, N)` from `dl_set.pt`
- Statistical models fit on train series `(T, N)` from `statsmodel_set.pt`
- Generated samples are stitched to reach target generation lengths
- All taxonomy metrics (fidelity, diversity, efficiency, and stylized facts) are computed
- Results are printed in the console and saved to detailed JSON files in the results directory

#### Customizing runs

- `configs/dataset_cfgs.yaml`: Set paths for `dl_set.pt` and `statsmodel_set.pt`.

---

## Docker Troubleshooting

### View logs for a specific service

```bash
docker-compose logs -f generate-data
```

### Rebuild after code changes

```bash
docker-compose build base
docker-compose up
```

### Run a single service with a custom command

```bash
# Build the base image first
docker-compose build base

# Run with a specific python command
docker-compose run --rm generate-data python src/generation_scripts/generate_data.py --generation_length 52
```

### Clean up

```bash
# Stop all containers
docker-compose down

# Remove volumes (WARNING: deletes data!)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

---

## Project Structure

```
Unified-benchmark-for-SDGFTS-main/
  ├─ data/                       # Raw and preprocessed datasets
  ├─ notebooks/                  # Validate functionality of parts of the pipeline
  ├─ results/                    # Evaluation results (JSON files)
  ├─ evaluation_plots/           # Publication-ready plots (generated)
  ├─ src/
  │   ├─ models/                 # Generative model implementations
  │   ├─ taxonomies/
  │   │   ├─ diversity.py        # Diversity metrics (e.g., ICD, ED, DTW)
  │   │   ├─ efficiency.py       # Efficiency metrics (runtime, memory)
  │   │   ├─ fidelity.py         # Fidelity/feature metrics + Visualization (MDD, MD, SDD, KD, ACD, t-SNE, Distrib. Plots)
  │   │   └─ stylized_facts.py   # Stylized facts metrics (tails, autocorr, volatility)
  │   ├─ plot_statistics/        # Plotting functionality for evaluation results
  │   │   └─ evaluation_plotter.py  # Main plotting script (executable)
  │   ├─ utils/                  # Configs, display, math, evaluation classes, preprocessing, etc.
  │   │   └─ eval_plot_utils.py  # Utilities for evaluation plotting
  │   ├─ data_downloader.py      # Yahoo Finance download utility
  │   └─ data_preprocessing.py   # Builds dl_set.pt and statsmodel_set.pt
  ├─ configs/                    # Experiment and preprocessing config templates
  ├─ requirements.txt
  └─ README.md
```

---

## Supported Models

The benchmark supports a range of both traditional statistical models and modern deep learning approaches:

<details>
<summary><strong>Statistical Models</strong></summary>

- <kbd>Geometric Brownian Motion (GBM)</kbd>
- <kbd>Ornstein-Uhlenbeck (OU) Process</kbd>
- <kbd>Merton Jump Diffusion (MJD)</kbd>
- <kbd>Double Exponential Jump Diffusion (DEJD)</kbd>
- <kbd>GARCH(1,1)</kbd>
- <kbd>Block Bootstrap</kbd>

</details>

<details>
<summary><strong>Deep Learning Models</strong></summary>

- <kbd>TimeGAN</kbd>
- <kbd>QuantGAN</kbd>
- <kbd>TimeVAE</kbd>
- <kbd>Sig-WGAN</kbd>

</details>

> All models share a unified interface for training, sample generation, and comprehensive metric evaluation.

---

## Metrics & Evaluation

### 1. Fidelity Metrics
- **Feature-based Distances**
  - Marginal Distribution Difference (MDD)
  - Mean Difference (MD)
  - Standard Deviation Difference (SDD)
  - Kurtosis Difference (KD)
  - AutoCorrelation Difference (ACD)
- **Visualization**
  - t-SNE Visualization
  - Distribution Comparison Plots

### 2. Diversity Metrics
- **Intra-Class Distance**
  - Euclidean Distance (ED)
  - Dynamic Time Warping (DTW)

### 3. Efficiency Metrics
- **Generation Time** (seconds for generating 500 samples)

### 4. Stylized Facts Metrics
- **Heavy Tails (Excess Kurtosis)**
- **Lag-1 Autocorrelation of Returns**
- **Volatility Clustering**
- **Long Memory in Volatility**
- **Non-Stationarity Detection**

Refer to `src/taxonomies/` for implementation details and to `src/utils/` for utility functions.

---

## How To Add Your Own Model

1. Implement your model in `src/models/statistical/` or `src/models/deep_learning/` and inherit from `StatisticalModel` or `DeepLearningModel`.
2. Register your model in `notebooks/pipeline_validation.py` by specifying it under `run_complete_evaluation`.
3. Rerun the pipeline and review your results in the `results/` directory!

---

## Results

All results are available in:
- The console (summary tables per model)
- `results/` directory (will be created with JSON results containing all metrics, parameters, and evaluation outputs)

---

## Contributors

| Name                  | Role                                 | Email                             |
|-----------------------|--------------------------------------|-----------------------------------|
| **Eddison Pham**      | Machine Learning Researcher & Engineer | eddison.pham@mail.utoronto.ca     |
| **Albert Lam Ho**     | Quantitative Researcher              | uyenlam.ho@mail.utoronto.ca       |
| **Yiqing Irene Huang**| Research Supervisor/Professor        | iy.huang@mail.utoronto.ca         |

---

## More

- For detailed examples and model-by-model usage, see `notebooks/`.
- To report issues or contribute, see the **Contributing** section below.

---