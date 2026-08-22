"""Evaluation classes for the taxonomy metrics."""

import multiprocessing
import os
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
    calculate_cmd,
    calculate_dcor_diff,
    visualize_tsne,
    visualize_distribution,
    visualize_qq,
    visualize_per_channel,
)
from src.taxonomies.stylized_facts import (
    autocorr_returns,
    volatility_clustering,
    long_memory_volatility,
)
from src.taxonomies.utility import (
    AugmentedTestingEvaluator,
)

__all__ = [
    "TaxonomyEvaluator",
    "DiversityEvaluator",
    "FidelityEvaluator",
    "StylizedFactsEvaluator",
    "VisualAssessmentEvaluator",
    "UtilityEvaluator",
    "PortfolioEvaluator",
    "PnLEvaluatorWrapper",
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


def _eval_workers() -> int:
    """Number of parallel channel workers from env, or 0 for sequential."""
    try:
        return max(0, int(os.environ.get("STONKBENCH_EVAL_WORKERS", "0") or 0))
    except ValueError:
        return 0


def _parallel_channel_map(worker_fn, jobs: list, n_workers: int) -> list:
    """Evaluate independent per-channel jobs in a process pool.

    Uses the ``spawn`` context so each worker is a fresh process with its own
    CPU-time accounting (login-node ``ulimit -t`` caps are per-process) and no
    inherited CUDA context. Falls back to sequential evaluation if the pool
    cannot be created (e.g. oversubscription).
    """
    ctx = multiprocessing.get_context("spawn")
    try:
        pool = ctx.Pool(processes=max(1, min(n_workers, len(jobs))))
        try:
            return pool.map(worker_fn, jobs)
        finally:
            # terminate() + join() avoids the occasional pool.join() hang seen
            # when a worker lingers in teardown; map() has already collected
            # every result so killing idle workers is safe.
            pool.terminate()
            pool.join()
    except Exception:
        return [worker_fn(job) for job in jobs]


def _diversity_channel_worker(args) -> dict:
    """Compute ICD (euclidean + dtw) for a single channel in a worker process."""
    channel, metrics = args
    return {f"icd_{m}": calculate_icd(channel, metric=m) for m in metrics}


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
        channels = _split_channels(syn)
        n_workers = _eval_workers()
        if n_workers >= 2 and len(channels) > 1:
            # Parallelize the (expensive, pure-Python DTW) per-channel ICD
            # computation across worker processes: identical results, much
            # lower wall time and per-process CPU usage.
            jobs = [(channel, metrics) for channel in channels]
            channel_results = _parallel_channel_map(_diversity_channel_worker, jobs, n_workers)
        else:
            channel_results = [
                {f"icd_{m}": calculate_icd(channel, metric=m) for m in metrics}
                for channel in channels
            ]
        if len(channel_results) == 1:
            self.results = channel_results[0]
        else:
            summary = _aggregate_channel_metrics(channel_results)
            self.results = {k: v["mean"] for k, v in summary.items()}
            self.results["per_channel"] = channel_results
        return self.results


class FidelityEvaluator(TaxonomyEvaluator):
    def evaluate(self) -> Dict[str, np.ndarray]:
        """Evaluate all fidelity metrics including cross-channel (multivariate) metrics.

        Per-channel marginal metrics (mdd, md, sdd, sd, kd) are computed channel-wise
        and aggregated to a mean when C > 1. Cross-channel metrics (cmd, dcor_diff)
        operate on all channels jointly and are only reported when C > 1.
        """
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

        # --- Cross-channel (multivariate) fidelity metrics ---
        num_channels = len(ori_channels)
        if num_channels > 1:
            try:
                self.results["cmd"] = calculate_cmd(self.ori_data, self.syn_data)
                self.results["dcor_diff"] = calculate_dcor_diff(self.ori_data, self.syn_data)
            except Exception as exc:
                print(f"Warning: Cross-channel fidelity metrics failed: {exc}")

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
    def __init__(self, ori_data: np.ndarray, syn_data: np.ndarray, results_dir: Path,
                 channel_names: list | None = None):
        super().__init__(ori_data, syn_data)
        self.results_dir = results_dir
        self.channel_names = channel_names

    def evaluate(self):
        model_results_dir = self.results_dir / "visualizations"
        model_results_dir.mkdir(parents=True, exist_ok=True)
        per_channel_dir = self.results_dir / "per_asset"
        per_channel_dir.mkdir(parents=True, exist_ok=True)

        # Each visualization is independent: a failure in one (e.g. t-SNE on
        # tiny sample sets) must never prevent the QQ / distribution / per-asset
        # plots from being written.
        steps = [
            ("tsne", lambda: visualize_tsne(self.ori_data, self.syn_data, str(model_results_dir))),
            ("distribution", lambda: visualize_distribution(self.ori_data, self.syn_data, str(model_results_dir))),
            ("qq", lambda: visualize_qq(
                self.ori_data, self.syn_data, str(model_results_dir),
                channel_names=self.channel_names, per_asset_dir=str(per_channel_dir))),
            ("per_channel", lambda: visualize_per_channel(
                self.ori_data, self.syn_data, str(per_channel_dir),
                channel_names=self.channel_names)),
        ]
        for name, fn in steps:
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"Warning: {name} visualization failed: {e}")


class UtilityEvaluator(TaxonomyEvaluator):
    """Utility-based evaluation for deep hedging models (augmented testing only)."""

    def __init__(
        self,
        real_train_log_returns: torch.Tensor,
        real_val_log_returns: torch.Tensor,
        synthetic_train_log_returns: torch.Tensor,
        real_train_initial: torch.Tensor,
        real_val_initial: torch.Tensor,
        synthetic_train_initial: torch.Tensor | None = None,
        seq_length: int | None = None,
        num_epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.real_train_log_returns = real_train_log_returns
        self.real_val_log_returns = real_val_log_returns
        self.synthetic_train_log_returns = synthetic_train_log_returns
        self.real_train_initial = real_train_initial
        self.real_val_initial = real_val_initial
        self.synthetic_train_initial = synthetic_train_initial
        self.seq_length = seq_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def evaluate(self) -> Dict[str, Any]:
        """Run augmented testing evaluation only."""
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

        self.results = {
            "augmented_testing": augmented_results,
        }

        print("[UtilityEvaluator] Utility evaluation complete.")
        return self.results


class PortfolioEvaluator(TaxonomyEvaluator):
    """Portfolio optimization downstream evaluation (DeMiguel et al. 2009)."""

    def __init__(
        self,
        real_test_log_returns: torch.Tensor,
        synthetic_log_returns: torch.Tensor,
        estimation_window: int = 252,
        rebalance_freq: int = 21,
        periods_per_year: int = 252,
        rf: float = 0.0,
        num_assets_ablation: list[int] | None = None,
    ):
        super().__init__()
        self.real_test_log_returns = real_test_log_returns
        self.synthetic_log_returns = synthetic_log_returns
        self.estimation_window = estimation_window
        self.rebalance_freq = rebalance_freq
        self.periods_per_year = periods_per_year
        self.rf = rf
        self.num_assets_ablation = num_assets_ablation

    def evaluate(self) -> Dict[str, Any]:
        from src.taxonomies.portfolio import PortfolioOptimizationEvaluator

        real = self.real_test_log_returns.cpu().numpy() if isinstance(self.real_test_log_returns, torch.Tensor) else self.real_test_log_returns
        synth = self.synthetic_log_returns.cpu().numpy() if isinstance(self.synthetic_log_returns, torch.Tensor) else self.synthetic_log_returns

        evaluator = PortfolioOptimizationEvaluator(
            real_test_returns=real,
            synthetic_returns=synth,
            estimation_window=self.estimation_window,
            rebalance_freq=self.rebalance_freq,
            periods_per_year=self.periods_per_year,
            rf=self.rf,
            num_assets_ablation=self.num_assets_ablation,
        )
        try:
            self.results = evaluator.evaluate()
        except Exception as e:
            print(f"Warning: Portfolio evaluation failed: {e}")
            self.results = {"error": str(e)}
        return self.results


class PnLEvaluatorWrapper(TaxonomyEvaluator):
    """P&L downstream evaluation wrapper."""

    def __init__(
        self,
        real_log_returns: torch.Tensor,
        synthetic_log_returns: torch.Tensor,
        strategy: str = "equal_weight_bh",
        periods_per_year: int = 252,
        initial_value: float = 1.0,
    ):
        super().__init__()
        self.real_log_returns = real_log_returns
        self.synthetic_log_returns = synthetic_log_returns
        self.strategy = strategy
        self.periods_per_year = periods_per_year
        self.initial_value = initial_value

    def evaluate(self) -> Dict[str, Any]:
        from src.taxonomies.pnl import PnLEvaluator

        real = self.real_log_returns.cpu().numpy() if isinstance(self.real_log_returns, torch.Tensor) else self.real_log_returns
        synth = self.synthetic_log_returns.cpu().numpy() if isinstance(self.synthetic_log_returns, torch.Tensor) else self.synthetic_log_returns

        evaluator = PnLEvaluator(
            real_log_returns=real,
            synthetic_log_returns=synth,
            strategy=self.strategy,
            periods_per_year=self.periods_per_year,
            initial_value=self.initial_value,
        )
        try:
            self.results = evaluator.evaluate()
        except Exception as e:
            print(f"Warning: P&L evaluation failed: {e}")
            self.results = {"error": str(e)}
        return self.results
