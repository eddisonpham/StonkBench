"""
Evaluation classes for the taxonomy metrics.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Callable
import numpy as np
import mlflow
import time

from src.taxonomies.diversity import calculate_icd
from src.taxonomies.fidelity import (
    calculate_mdd, calculate_md, calculate_sdd, calculate_sd, calculate_kd, calculate_acd, visualize_tsne, visualize_distribution
)
from src.taxonomies.stylized_facts import (
    heavy_tails, autocorr_raw, volatility_clustering, long_memory_abs, non_stationarity
)


class TaxonomyEvaluator(ABC):
    """Abstract base class for taxonomy evaluators."""

    def __init__(self, ori_data: np.ndarray = None, syn_data: np.ndarray = None):
        self.ori_data = ori_data
        self.syn_data = syn_data
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation and return a dictionary of metrics."""
        pass

    def get_results(self) -> Dict[str, Any]:
        return self.results

class DiversityEvaluator(TaxonomyEvaluator):
    def evaluate(self) -> Dict[str, float]:
        metrics = ["euclidean", "dtw"]
        self.results = {f"icd_{m}": calculate_icd(self.syn_data, metric=m) for m in metrics}
        return self.results

class FidelityEvaluator(TaxonomyEvaluator):
    def evaluate(self) -> Dict[str, float]:
        fidelity_metrics = {
            "mdd": calculate_mdd,
            "md": calculate_md,
            "sdd": calculate_sdd,
            "sd": calculate_sd,
            "kd": calculate_kd,
            "acd": calculate_acd
        }
        self.results = {name: fn(self.ori_data, self.syn_data) for name, fn in fidelity_metrics.items()}
        return self.results

class RuntimeEvaluator(TaxonomyEvaluator):
    """
    Evaluates the runtime of a synthetic data generation function.
    """
    def __init__(self, generate_func: Callable, num_samples: int = 500, generation_kwargs: Dict[str, Any] = None):
        super().__init__()
        self.generate_func = generate_func
        self.num_samples = num_samples
        self.generation_kwargs = generation_kwargs or {}

    def evaluate(self) -> Dict[str, float]:
        start_time = time.perf_counter()
        _ = self.generate_func(self.num_samples, **self.generation_kwargs)
        end_time = time.perf_counter()
        runtime = round(end_time - start_time, 4)
        self.results = {f"generation_time_{self.num_samples}_samples": runtime}
        return self.results

class StylizedFactsEvaluator(TaxonomyEvaluator):
    def evaluate(self) -> Dict[str, Any]:

        fact_functions = {
            "heavy_tails": heavy_tails,
            "autocorr_raw": autocorr_raw,
            "volatility_clustering": volatility_clustering,
            "long_memory": long_memory_abs,
            "non_stationarity": non_stationarity
        }

        try:
            for name, fn in fact_functions.items():
                real_val = fn(self.ori_data)
                synth_val = fn(self.syn_data)
                self.results[f"{name}_real"] = real_val.tolist()
                self.results[f"{name}_synth"] = synth_val.tolist()
                self.results[f"{name}_diff"] = np.abs(real_val - synth_val).tolist()
        except Exception as e:
            print(f"Warning: Stylized facts evaluation failed: {e}")
            self.results["stylized_facts_error"] = str(e)

        return self.results

class VisualAssessmentEvaluator(TaxonomyEvaluator):
    def __init__(self, ori_data: np.ndarray, syn_data: np.ndarray, results_dir: Path, timestamp: str):
        super().__init__(ori_data, syn_data)
        self.results_dir = results_dir
        self.timestamp = timestamp

    def evaluate(self, model_name: str):
        try:
            model_results_dir = self.results_dir / f"visualizations_{model_name}"
            model_results_dir.mkdir(parents=True, exist_ok=True)

            visualize_tsne(self.ori_data, self.syn_data, str(model_results_dir), model_name)
            visualize_distribution(self.ori_data, self.syn_data, str(model_results_dir), model_name)

            mlflow.log_artifacts(str(model_results_dir))
        except Exception as e:
            print(f"Warning: Visual assessment failed: {e}")
