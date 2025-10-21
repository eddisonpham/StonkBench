# Unified Benchmark for Synthetic Data Generation in Financial Time Series (SDGFTS)

> A unified, reproducible benchmark for evaluating synthetic time series generators in finance. All results, metrics, and experiment outputs are automatically tracked and organized using [MLFlow](https://mlflow.org/).

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

### 3. ▶️ Run the Benchmark

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
  - Tracked as an experiment in **MLFlow** with all parameters, scores, and output artifacts.

#### Customizing runs:
- `configs/dataset_cfgs.yaml`: Modify the preprocessing of the dataset for *parametric/non-parametric*.
- `configs/model_cfgs.yaml`: Configurations for deep learning models (or non-parametric).


### 4. 📊 Viewing Results in MLFlow

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

## 🗂️ Project Structure

```
Unified-benchmark-for-SDGFTS-main/
  ├─ data/                       # Raw and preprocessed datasets
  ├─ notebooks/                  # Validate functionality of parts of the pipeline
  ├─ src/
  │   ├─ models/                 # Generative model implementations
  │   ├─ taxonomies/
  │   │   ├─ diversity.py        # Diversity metrics (e.g., ICD, ED, DTW)
  │   │   ├─ efficiency.py       # Efficiency metrics (runtime, memory)
  │   │   ├─ fidelity.py         # Fidelity/feature metrics + Visualization (MDD, MD, SDD, KD, ACD, t-SNE, Distrib. Plots)
  │   │   └─ stylized_facts.py   # Stylized facts metrics (tails, autocorr, volatility)
  │   ├─ utils/                  # Configs, display, math, evaluation classes, preprocessing, etc.
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
3. Rerun the pipeline and review your new runs in MLFlow!

---

## 🏆 Results

All results are available in:
- The console (summary tables per model)
- `results/` directory (will be created with JSON results)
- **MLFlow UI** (`mlruns/` directory, browsable at [http://localhost:5000](http://localhost:5000)) — all metrics, parameters, and artifacts are logged automatically.

---

## 👥 Contributors

| Name                  | Role                                 | Email                             |
|-----------------------|--------------------------------------|-----------------------------------|
| **Eddison Pham**      | Machine Learning Researcher/Engineer | eddison.pham@mail.utoronto.ca     |
| **Albert Lam Ho**     | Quantitative Researcher              | uyenlam.ho@mail.utoronto.ca       |
| **Yiqing Irene Huang**| Research Supervisor/Professor        | iy.huang@mail.utoronto.ca         |

---

## 📚 More

- For detailed examples and model-by-model usage, see `notebooks/`.
- To report issues or contribute, see the **Contributing** section below.

---

