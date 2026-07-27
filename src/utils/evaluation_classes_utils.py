"""Evaluation classes for the taxonomy metrics."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from src.taxonomies.diversity import calculate_icd
from src.taxonomies.fidelity import (
    calculate_mdd,
    calculate_md,
    calculate_sdd,
    calculate_sd,
    calculate_kd,
    visualize_tsne,
    visualize_distribution,
)
from src.taxonomies.stylized_facts import (
    autocorr_returns,
    volatility_clustering,
    long_memory_volatility,
)
from src.taxonomies.utility import (
    AugmentedTestingEvaluator,
    AlgorithmComparisonEvaluator,
)

__all__ = [
    "TaxonomyEvaluator",
    "DiversityEvaluator",
    "FidelityEvaluator",
    "StylizedFactsEvaluator",
    "VisualAssessmentEvaluator",
    "UtilityEvaluator",
]


def _to_2d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        n, l, c = arr.shape
        return arr.reshape(n, l * c)
    raise ValueError(f"Expected 2D or 3D data, got shape {arr.shape}")


def _split_channels(data: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(data)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [arr[:, :, c] for c in range(arr.shape[2])]
    raise ValueError(f"Expected 2D or 3D data, got shape {arr.shape}")


def _aggregate_channel_metrics(channel_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not channel_metrics:
        return {}
    keys = sorted(channel_metrics[0].keys())
    aggregated = {}
    for key in keys:
        values = np.array([m[key] for m in channel_metrics], dtype=float)
        aggregated[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return aggregated


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
        syn = np.asarray(self.syn_data)
        channel_results = []
        for channel in _split_channels(syn):
            channel_results.append({f"icd_{m}": calculate_icd(channel, metric=m) for m in metrics})
        if len(channel_results) == 1:
            self.results = channel_results[0]
        else:
            summary = _aggregate_channel_metrics(channel_results)
            self.results = {k: v["mean"] for k, v in summary.items()}
            self.results["per_channel"] = channel_results
        return self.results


class FidelityEvaluator(TaxonomyEvaluator):
    def evaluate(self) -> Dict[str, np.ndarray]:
        fidelity_metrics = {
            "mdd": calculate_mdd,
            "md": calculate_md,
            "sdd": calculate_sdd,
            "sd": calculate_sd,
            "kd": calculate_kd,
        }
        ori_channels = _split_channels(np.asarray(self.ori_data))
        syn_channels = _split_channels(np.asarray(self.syn_data))
        if len(ori_channels) != len(syn_channels):
            min_c = min(len(ori_channels), len(syn_channels))
            ori_channels = ori_channels[:min_c]
            syn_channels = syn_channels[:min_c]

        channel_results = []
        for ori_c, syn_c in zip(ori_channels, syn_channels):
            channel_results.append({name: fn(ori_c, syn_c) for name, fn in fidelity_metrics.items()})

        if len(channel_results) == 1:
            self.results = channel_results[0]
        else:
            summary = _aggregate_channel_metrics(channel_results)
            self.results = {k: v["mean"] for k, v in summary.items()}
            self.results["per_channel"] = channel_results
        return self.results


class StylizedFactsEvaluator(TaxonomyEvaluator):
    def evaluate(self) -> Dict[str, Any]:
        fact_functions = {
            "autocorr_returns": autocorr_returns,
            "volatility_clustering": volatility_clustering,
            "long_memory_volatility": long_memory_volatility,
        }
        try:
            ori_channels = _split_channels(np.asarray(self.ori_data))
            syn_channels = _split_channels(np.asarray(self.syn_data))
            per_channel = []
            for ori_c, syn_c in zip(ori_channels, syn_channels):
                channel_dict = {}
                for name, fn in fact_functions.items():
                    real_val = fn(ori_c)
                    synth_val = fn(syn_c)
                    diff_val = np.abs(real_val - synth_val)
                    channel_dict[name] = {
                        "real": float(np.asarray(real_val).mean()),
                        "synth": float(np.asarray(synth_val).mean()),
                        "diff": float(np.asarray(diff_val).mean()),
                    }
                per_channel.append(channel_dict)
            if len(per_channel) == 1:
                self.results = per_channel[0]
            else:
                averaged = {}
                for metric_name in per_channel[0].keys():
                    real_vals = np.array([c[metric_name]["real"] for c in per_channel], dtype=float)
                    synth_vals = np.array([c[metric_name]["synth"] for c in per_channel], dtype=float)
                    diff_vals = np.array([c[metric_name]["diff"] for c in per_channel], dtype=float)
                    averaged[metric_name] = {
                        "real": float(np.mean(real_vals)),
                        "synth": float(np.mean(synth_vals)),
                        "diff": float(np.mean(diff_vals)),
                    }
                averaged["per_channel"] = per_channel
                self.results = averaged
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
            model_results_dir = self.results_dir / "visualizations"
            model_results_dir.mkdir(parents=True, exist_ok=True)

            visualize_tsne(self.ori_data, self.syn_data, str(model_results_dir))
            visualize_distribution(self.ori_data, self.syn_data, str(model_results_dir))
        except Exception as e:
            print(f"Warning: Visual assessment failed: {e}")


class UtilityEvaluator(TaxonomyEvaluator):
    """Utility-based evaluation for deep hedging models."""

    def __init__(
        self,
        real_train_log_returns: torch.Tensor,
        real_val_log_returns: torch.Tensor,
        real_test_log_returns: torch.Tensor,
        synthetic_train_log_returns: torch.Tensor,
        synthetic_val_log_returns: torch.Tensor,
        synthetic_test_log_returns: torch.Tensor,
        real_train_initial: torch.Tensor,
        real_val_initial: torch.Tensor,
        real_test_initial: torch.Tensor,
        synthetic_train_initial: torch.Tensor | None = None,
        synthetic_val_initial: torch.Tensor | None = None,
        synthetic_test_initial: torch.Tensor | None = None,
        seq_length: int | None = None,
        num_epochs: int = 2,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.real_train_log_returns = real_train_log_returns
        self.real_val_log_returns = real_val_log_returns
        self.real_test_log_returns = real_test_log_returns
        self.synthetic_train_log_returns = synthetic_train_log_returns
        self.synthetic_val_log_returns = synthetic_val_log_returns
        self.synthetic_test_log_returns = synthetic_test_log_returns
        self.real_train_initial = real_train_initial
        self.real_val_initial = real_val_initial
        self.real_test_initial = real_test_initial
        self.synthetic_train_initial = synthetic_train_initial
        self.synthetic_val_initial = synthetic_val_initial
        self.synthetic_test_initial = synthetic_test_initial
        self.seq_length = seq_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def evaluate(self) -> Dict[str, Any]:
        """Run both augmented testing and algorithm comparison evaluations."""
        print("[UtilityEvaluator] Starting utility evaluation...")

        augmented_evaluator = AugmentedTestingEvaluator(
            real_train_log_returns=self.real_train_log_returns,
            real_val_log_returns=self.real_val_log_returns,
            synthetic_train_log_returns=self.synthetic_train_log_returns,
            real_train_initial=self.real_train_initial,
            real_val_initial=self.real_val_initial,
            synthetic_train_initial=self.synthetic_train_initial,
            seq_length=self.seq_length,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

        try:
            augmented_results = augmented_evaluator.evaluate()
        except Exception as e:
            print(f"Warning: Augmented testing evaluation failed: {e}")
            augmented_results = {"error": str(e)}

        algorithm_evaluator = AlgorithmComparisonEvaluator(
            real_train_log_returns=self.real_train_log_returns,
            real_test_log_returns=self.real_test_log_returns,
            synthetic_train_log_returns=self.synthetic_train_log_returns,
            real_train_initial=self.real_train_initial,
            real_test_initial=self.real_test_initial,
            synthetic_train_initial=self.synthetic_train_initial,
            seq_length=self.seq_length,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

        try:
            algorithm_comparison_results = algorithm_evaluator.evaluate()
        except Exception as e:
            print(f"Warning: Algorithm comparison evaluation failed: {e}")
            algorithm_comparison_results = {"error": str(e)}

        self.results = {
            "augmented_testing": augmented_results,
            "algorithm_comparison": algorithm_comparison_results,
        }

        print("[UtilityEvaluator] Utility evaluation complete.")
        return self.results
