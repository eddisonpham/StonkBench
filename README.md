# 🚀 Unified Benchmark for Synthetic Data Generation in Financial Time Series (SDGFTS)

> A unified, reproducible benchmark for evaluating synthetic time series generators in finance. All results, metrics, and experiment outputs are automatically tracked and organized using [MLFlow](https://mlflow.org/).

---

## ⚡ Quickstart

### 1. 🛠️ Installation

- Python: 3.9 or newer (recommended)
- Install all dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. 📥 Download Dataset

Fetch the required (Google stock, 5 years daily) dataset:
```bash
python src/data_downloader.py
```

This will save the data as `data/raw/GOOG/GOOG.csv`.

### 3. ▶️ Run the Benchmark

Execute the full benchmark and get all evaluation metrics, synthetic data, and logs:
```bash
python src/evaluator.py
```

**What happens:**
- ⚙️ **Data Preprocessing**:
  - **Non-parametric models**: The data is segmented into overlapping sub-sequences of shape `(R, l, N)` where `R` is the number of sequences, `l` is the sequence length, and `N` is the number of features.
  - **Parametric models**: The original time series is used without segmentation, resulting in data of shape `(l, N)`.
- 🤖 Several generative models (both parametric and non-parametric) are trained.
- 🧬 Each model generates exactly **500 samples**.
- 📊 All taxonomy metrics (fidelity, diversity, efficiency, and stylized facts) are computed.
- Results are:
  - 🖥️ Printed in the console.
  - 📝 Saved to a detailed JSON file in the results directory.
  - 📦 Tracked as an experiment in **MLFlow** with all parameters, scores, and output artifacts.

#### Customizing runs:
- ✍️ Edit `dataset_config` and `models_config` dictionaries in [`src/evaluator.py`](src/evaluator.py) to change paths, sample counts, model parameters, etc.

### 4. 📊 Viewing Results in MLFlow

After you run the benchmark, use MLFlow’s UI to explore and compare your experiments:

1. Start the MLFlow tracking UI (in your project root):
   ```bash
   mlflow ui
   ```
2. Visit [http://localhost:5000](http://localhost:5000) in your browser.
3. For each experiment/model, you’ll see:
   - 📁 Parameters/configurations
   - ⏱️ Training time, generation time (for 500 samples)
   - 📈 All computed metrics (Fidelity, Diversity, Efficiency, Stylized Facts)
   - 📎 Downloadable output artifacts (e.g., metrics JSON, visualization plots)
4. Use MLFlow to compare models across any metric, check plots, and download results.

---

## 🗂️ Project Structure

```
Unified-benchmark-for-SDGFTS-main/
  ├─ data/                       # Raw and preprocessed datasets
  ├─ notebooks/                  # Interactive explorations, validation, test runs
  ├─ src/
  │   ├─ models/                 # Generative model implementations
  │   ├─ preprocessing/          # Data preprocessing and transformations
  │   ├─ taxonomies/
  │   │   ├─ diversity.py        # Diversity metrics (e.g., ICD, ED, DTW)
  │   │   ├─ efficiency.py       # Efficiency metrics (runtime, memory)
  │   │   ├─ fidelity.py         # Fidelity/feature metrics (MDD, MD, SDD, KD, ACD, etc.)
  │   │   └─ stylized_facts.py   # Stylized facts metrics (tails, autocorr, volatility)
  │   ├─ utils/                  # Utility modules, IO, math, paths, etc.
  │   ├─ data_downloader.py      # Dataset download utility
  │   └─ evaluator.py            # Main pipeline and evaluation runner
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

- <kbd>Vanilla GAN</kbd>
- <kbd>Wasserstein GAN</kbd>
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
- **Memory Usage** (peak MB during generation)

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
2. Register your model in `src/evaluator.py` under the `models` dictionary in `run_complete_evaluation`.
3. Rerun the pipeline and review your new runs in MLFlow!

---

## 🏆 Results

All results are available in:
- 🖥️ The console (summary tables per model)
- 📁 `data/evaluation_results/` directory (detailed JSON for each run)
- 📊 **MLFlow UI** (`mlruns/` directory, browsable at [http://localhost:5000](http://localhost:5000)) — all metrics, parameters, and artifacts are logged automatically.

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

