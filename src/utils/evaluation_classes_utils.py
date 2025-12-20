"""
Evaluation classes for the taxonomy metrics.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Callable, Optional
import numpy as np
import torch
import time

from src.taxonomies.diversity import calculate_icd
from src.taxonomies.fidelity import (
    calculate_mdd, calculate_md, calculate_sdd, calculate_sd, calculate_kd, visualize_tsne, visualize_distribution
)
from src.taxonomies.stylized_facts import (
    autocorr_returns, volatility_clustering, long_memory_volatility
)
from src.taxonomies.utility import (
    AugmentedTestingEvaluator,
    AlgorithmComparisonEvaluator
)

from src.hedging_models.deep_hedgers.feedforward_layers import FeedforwardLayers
from src.hedging_models.deep_hedgers.feedforward_time import FeedforwardTime
from src.hedging_models.deep_hedgers.rnn_hedger import RNN
from src.hedging_models.deep_hedgers.lstm_hedger import LSTM
from src.hedging_models.non_deep_hedgers.black_scholes import BlackScholes
from src.hedging_models.non_deep_hedgers.delta_gamma import DeltaGamma
from src.hedging_models.non_deep_hedgers.linear_regression import LinearRegression
from src.hedging_models.non_deep_hedgers.xgboost import XGBoost


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
            "autocorr_returns": autocorr_returns,
            "volatility_clustering": volatility_clustering,
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

class UtilityEvaluator(TaxonomyEvaluator):
    """
    Utility-based evaluation for deep hedging models.
    Evaluates synthetic data quality using two methods:
    1. Augmented Testing: Mix synthetic with real training data (50/50), train hedger, compare with real-only
    2. Algorithm Comparison: Train hedgers on both real and synthetic data, evaluate on test sets
    """
    
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
        synthetic_train_initial: Optional[torch.Tensor] = None,
        synthetic_val_initial: Optional[torch.Tensor] = None,
        synthetic_test_initial: Optional[torch.Tensor] = None,
        seq_length: Optional[int] = None,
        num_epochs: int = 2,
        batch_size: int = 64,
        learning_rate: float = 1e-3
    ):
        """
        Initialize utility evaluator.
        
        Args:
            real_train_log_returns: Real training log returns (R_train, L)
            real_val_log_returns: Real validation log returns (R_val, L)
            real_test_log_returns: Real test log returns (R_test, L)
            synthetic_train_log_returns: Synthetic training log returns (R_syn_train, L)
            synthetic_val_log_returns: Synthetic validation log returns (R_syn_val, L)
            synthetic_test_log_returns: Synthetic test log returns (R_syn_test, L)
            real_train_initial: Real training initial prices (R_train,)
            real_val_initial: Real validation initial prices (R_val,)
            real_test_initial: Real test initial prices (R_test,)
            synthetic_train_initial: Synthetic training initial prices (R_syn_train,)
            synthetic_val_initial: Synthetic validation initial prices (R_syn_val,)
            synthetic_test_initial: Synthetic test initial prices (R_syn_test,)
            seq_length: Sequence length (inferred if None)
            num_epochs: Number of training epochs for hedgers
            batch_size: Batch size for hedger training
            learning_rate: Learning rate for hedger training
        """
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
        """
        Run both augmented testing and algorithm comparison evaluations.
        
        Returns:
            Dictionary with 'augmented_testing' and 'algorithm_comparison' results
        """
        print("[UtilityEvaluator] Starting utility evaluation...")
        
        # Run Augmented Testing Evaluation
        print("[UtilityEvaluator] Running Augmented Testing Evaluation...")
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
            learning_rate=self.learning_rate
        )

        try:
            augmented_results = augmented_evaluator.evaluate()
        except Exception as e:
            print(f"Warning: Augmented testing evaluation failed: {e}")
            augmented_results = {"error": str(e)}
        
        # Run Algorithm Comparison Evaluation
        print("[UtilityEvaluator] Running Algorithm Comparison Evaluation...")
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
            learning_rate=self.learning_rate
        )

        
        try:
            algorithm_comparison_results = algorithm_evaluator.evaluate()
        except Exception as e:
            print(f"Warning: Algorithm comparison evaluation failed: {e}")
            algorithm_comparison_results = {"error": str(e)}
        
        self.results = {
            "augmented_testing": augmented_results,
            "algorithm_comparison": algorithm_comparison_results
        }
        
        print("[UtilityEvaluator] Utility evaluation complete.")
        return self.results
