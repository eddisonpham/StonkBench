# Unified Benchmark for Synthetic Data Generation in Financial Time Series (SDGFTS)

> A unified, reproducible benchmark for evaluating synthetic time series generators in finance. All results, metrics, and experiment outputs are automatically saved and organized.

---

## ⚡ Quickstart

### 1. Installation

- Python: 3.9 or newer (recommended)
- Install all dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. 📥 Download Dataset

Fetch the required dataset:
```bash
python src/data_downloader.py --ticker `[ticker name]`
```

This will save the data as `data/raw/[ticker name]/[ticker name].csv`.

### 3. 🐳 Run with Docker (optional)

Build runtime image:
```bash
docker build -f Dockerfile.runtime -t sdgfts-runtime .
```

Generate artifacts (volume mount your working directory):
```bash
docker run --rm -v $(pwd):/app sdgfts-runtime \
  python src/scripts/generate_parametric_data.py --seq_lengths 120 180 --num_samples 500
docker run --rm -v $(pwd):/app sdgfts-runtime \
  python src/scripts/generate_non_parametric_data.py --seq_lengths 120 180 --num_samples 500
```

Evaluate saved artifacts:
```bash
docker run --rm -v $(pwd):/app sdgfts-runtime \
  python src/unified_evaluator.py --seq_lengths 120 180 --results_dir results/seq_run
```

Orchestrate all stages locally (non-container):
```bash
python src/parallelizer_script.py --seq_lengths 120 180 --stage all
```

### 4. ▶️ Run the Benchmark

Execute the full benchmark and get all evaluation metrics, synthetic data, and logs in `notebooks/pipeline_validation.py`


**What happens:**
- **Data Preprocessing**:
  - **Non-parametric models**: The data is segmented into overlapping sub-sequences of shape `(R, l, N)` where `R` is the number of sequences, `l` is the sequence length, and `N` is the number of features.
  - **Parametric models**: The original time series is used without segmentation, resulting in data of shape `(l, N)`.
- Several generative models (both parametric and non-parametric) are trained.
- Each model generates exactly **500 samples**.
- All taxonomy metrics (fidelity, diversity, efficiency, and stylized facts) are computed.
- Results are:
  -  Printed in the console.
  - Saved to a detailed JSON file in the results directory.

#### Customizing runs:
- `configs/dataset_cfgs.yaml`: Modify the preprocessing of the dataset for *parametric/non-parametric*.

### 4. 📊 Viewing Results

#### Publication-Ready Plots
Generate comprehensive, publication-ready plots for all evaluation metrics:

```bash
python src/plot_statistics/evaluation_plotter.py
```

This will:
- Automatically find the latest evaluation results
- Generate publication-quality plots (300 DPI) for all metrics
- Save plots to `evaluation_plots/` directory
- Include performance metrics, distribution analysis, similarity measures, stylized facts, and model rankings

---

## 🗂️ Project Structure

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
  │   └─ data_downloader.py      # Dataset download utility
  ├─ configs/                    # Experiment and preprocessing config templates
  ├─ requirements.txt
  └─ README.md
```

---

## 🤖 Supported Models

The benchmark supports a range of both traditional parametric models and modern deep learning approaches:

<details>
<summary><strong>Parametric Models</strong></summary>

- <kbd>Geometric Brownian Motion (GBM)</kbd>
- <kbd>Ornstein-Uhlenbeck (OU) Process</kbd>
- <kbd>Merton Jump Diffusion (MJD)</kbd>
- <kbd>Double Exponential Jump Diffusion (DEJD)</kbd>
- <kbd>GARCH(1,1)</kbd>

</details>

<details>
<summary><strong>Non-parametric & Deep Learning Models</strong></summary>

- <kbd>TimeGAN</kbd>
- <kbd>QuantGAN</kbd>
- <kbd>TimeVAE</kbd>
- <kbd>Sig-WGAN</kbd>
- <kbd>Block Bootstrap</kbd>

</details>

> 🛠️ All models share a unified interface for training, sample generation, and comprehensive metric evaluation.

---

## 📏 Metrics & Evaluation

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

## ➕ How To Add Your Own Model

1. Implement your model in `src/models/` and ensure you inherit from the appropriate base class (`ParametricModel` or `DeepLearningModel`).
2. Register your model in `notebooks/pipeline_validation.py` by specifying it under `run_complete_evaluation`.
3. Rerun the pipeline and review your results in the `results/` directory!

---

## 🏆 Results

All results are available in:
- The console (summary tables per model)
- `results/` directory (will be created with JSON results containing all metrics, parameters, and evaluation outputs)

---

## 👥 Contributors

| Name                  | Role                                 | Email                             |
|-----------------------|--------------------------------------|-----------------------------------|
| **Eddison Pham**      | Machine Learning Researcher & Engineer | eddison.pham@mail.utoronto.ca     |
| **Albert Lam Ho**     | Quantitative Researcher              | uyenlam.ho@mail.utoronto.ca       |
| **Yiqing Irene Huang**| Research Supervisor/Professor        | iy.huang@mail.utoronto.ca         |

---

## 📚 More

- For detailed examples and model-by-model usage, see `notebooks/`.
- To report issues or contribute, see the **Contributing** section below.

---

