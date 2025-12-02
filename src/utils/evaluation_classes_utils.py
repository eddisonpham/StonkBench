"""
Evaluation classes for the taxonomy metrics.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Callable
import numpy as np
import time

from src.taxonomies.diversity import calculate_icd
from src.taxonomies.fidelity import (
    calculate_mdd, calculate_md, calculate_sdd, calculate_sd, calculate_kd, visualize_tsne, visualize_distribution
)
from src.taxonomies.stylized_facts import (
    excess_kurtosis, autocorr_returns, volatility_clustering, leverage_effect, long_memory_volatility
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
    def evaluate(self) -> Dict[str, np.ndarray]:
        metrics = ["euclidean", "dtw"]
        self.results = {f"icd_{m}": calculate_icd(self.syn_data, metric=m) for m in metrics}
        return self.results

class FidelityEvaluator(TaxonomyEvaluator):
    def evaluate(self) -> Dict[str, np.ndarray]:
        fidelity_metrics = {
            "mdd": calculate_mdd,
            "md": calculate_md,
            "sdd": calculate_sdd,
            "sd": calculate_sd,
            "kd": calculate_kd
        }
        self.results = {name: fn(self.ori_data, self.syn_data) for name, fn in fidelity_metrics.items()}
        return self.results

class RuntimeEvaluator(TaxonomyEvaluator):
    """
    Evaluates the runtime of a synthetic data generation function.
    """
    def __init__(self, generate_func: Callable, generation_kwargs: Dict[str, Any] = None):
        super().__init__()
        self.generate_func = generate_func
        self.generation_kwargs = generation_kwargs or {}

    def evaluate(self) -> Dict[str, float]:
        num_samples = self.generation_kwargs.get('num_samples', 500)
        start_time = time.perf_counter()
        _ = self.generate_func(**self.generation_kwargs)
        end_time = time.perf_counter()
        runtime = round(end_time - start_time, 4)
        self.results = {f"generation_time_{num_samples}_samples": runtime}
        return self.results

class StylizedFactsEvaluator(TaxonomyEvaluator):
    def evaluate(self) -> Dict[str, Any]:
        fact_functions = {
            "excess_kurtosis": excess_kurtosis,
            "autocorr_returns": autocorr_returns,
            "volatility_clustering": volatility_clustering,
            "leverage_effect": leverage_effect,
            "long_memory_volatility": long_memory_volatility
        }
        try:
            for name, fn in fact_functions.items():
                real_val = fn(self.ori_data)
                synth_val = fn(self.syn_data)
                diff_val = np.abs(real_val - synth_val)
                # Store results as scalars
                self.results[name] = {
                    "real": float(real_val),
                    "synth": float(synth_val),
                    "diff": float(diff_val)
                }
        except Exception as e:
            print(f"Warning: Stylized facts evaluation failed: {e}")
            self.results["stylized_facts_error"] = str(e)

        return self.results

class VisualAssessmentEvaluator(TaxonomyEvaluator):
    def __init__(self, ori_data: np.ndarray, syn_data: np.ndarray, results_dir: Path):
        super().__init__(ori_data, syn_data)
        self.results_dir = results_dir

    def evaluate(self):
        try:
            model_results_dir = self.results_dir / f"visualizations"
            model_results_dir.mkdir(parents=True, exist_ok=True)

            visualize_tsne(self.ori_data, self.syn_data, str(model_results_dir))
            visualize_distribution(self.ori_data, self.syn_data, str(model_results_dir))
        except Exception as e:
            print(f"Warning: Visual assessment failed: {e}")
