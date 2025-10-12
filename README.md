# Unified Benchmark for Synthetic Data Generation in Financial Time Series (SDGFTS)

> A unified, reproducible benchmark for evaluating synthetic time series generators in finance. All results, metrics, and experiment outputs are automatically tracked and organized using [MLFlow](https://mlflow.org/).

---

## Quickstart

### 1. Installation

- Python: 3.9 or newer (recommended)
- Install all dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Download Dataset

Fetch the required (Google stock, 5 years daily) dataset:
```bash
python src/ingestion/data_downloader.py
```

This will save the data as `data/raw/GOOG/GOOG.csv`.

### 3. Run the Benchmark

Execute the full benchmark and get all evaluation metrics, synthetic data, and logs:
```bash
python src/experiments/evaluator.py
```

**What happens:**
- Data is preprocessed (see configs in `main()` of `src/experiments/evaluator.py`)
- Several generative models are trained
- Each model generates exactly **500 samples**
- All taxonomy metrics (fidelity, diversity, efficiency and utility) are computed.
- Results are:
  - Printed in the console
  - Saved to a detailed JSON file in the results directory
  - Tracked as an experiment in **MLFlow** with all parameters, scores, and output artifacts

#### Customizing runs:
- Edit `dataset_config` and `models_config` dictionaries at the top of [`src/experiments/evaluator.py`](src/experiments/evaluator.py) to change paths, sample counts, model parameters, etc.

### 4. Viewing Results in MLFlow

After you run the benchmark, use MLFlow’s UI to explore and compare your experiments:

1. Start the MLFlow tracking UI (in your project root):
   ```bash
   mlflow ui
   ```
2. Visit [http://localhost:5000](http://localhost:5000) in your browser.
3. For each experiment/model, you’ll see:
   - Parameters/configurations
   - Training time, generation time (for 500 samples)
   - All computed metrics (Fidelity, Diversity, Efficiency, Stylized Facts)
   - Downloadable output artifacts (e.g., metrics JSON, visualization plots)
4. Use MLFlow to compare models across any metric, check plots, and download results.

---

## Project Structure

```
Unified-benchmark-for-SDGFTS-main/
  ├─ data/                   # Raw and preprocessed datasets
  ├─ notebooks/              # Interactive explorations, validation, test runs
  ├─ src/
  │   ├─ experiments/        # Main pipeline and evaluation runner
  │   ├─ models/             # Generative model implementations
  │   ├─ preprocessing/      # Data preprocessing and transformations
  │   ├─ evaluation/         # Metric computation and visualization
  │   └─ utils/              # Utility modules, IO, math, paths, etc
  ├─ configs/                # Experiment and preprocessing config templates
  ├─ requirements.txt
  └─ README.md
```

---

## Supported Models

- **Parametric:** Geometric Brownian Motion (GBM), Ornstein-Uhlenbeck (OU) Process, Merton Jump Diffusion, Stochastic Volatility, GARCH(1, 1)
- **Non-parametric / Deep Learning:** Vanilla GAN, Wasserstein GAN

All models have a unified interface for training, generation, and metric evaluation.

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
- **Memory Usage** (peak MB during generation)

### 4. Stylized Facts Metrics
- **Heavy Tails (Excess Kurtosis)**
- **Lag-1 Autocorrelation of Returns**
- **Volatility Clustering**
- **Long Memory in Volatility**
- **Non-Stationarity Detection**

Refer to `src/evaluation/metrics/` for implementation details and to `src/evaluation/visualizations/plots.py` for visualization code.
---

## How To Add Your Own Model

1. Implement your model in `src/models/` and ensure you inherit from the appropriate base class.
2. Register your model in `src/experiments/evaluator.py` under the `models` dict in `run_complete_evaluation`.
3. Rerun the pipeline and review your new runs in MLFlow!

---

## Results

All results are available in:
- The console (summary tables per model)
- `results/` directory (detailed JSON for each run)
- **MLFlow UI** (`mlruns/` directory, browsable at [http://localhost:5000](http://localhost:5000)) — all metrics, parameters, and artifacts are logged automatically

---

## Contributors

- **Eddison Pham** (Machine Learning Researcher/Engineer, eddison.pham@mail.utoronto.ca)
- **Albert Lam Ho** (Quantitative Researcher, uyenlam.ho@mail.utoronto.ca)
- **Yiqing Irene Huang** (Research Supervisor/Professor, iy.huang@mail.utoronto.ca)

---

## More

- For detailed examples and model-by-model usage, see `notebooks/`.
- To report issues or contribute, see the **Contributing** section below.

---

## Citation, License, Contact, Contributing

TBD

---

