"""
Utility-based evaluation for deep hedging models.

This module implements two types of evaluations:
1. Augmented Testing: Mix synthetic with real training data (50/50), train hedger, compare with real-only
2. Algorithm Comparison: Generate train/val/test synthetic data, train 4 hedgers on both real and synthetic, evaluate

The scores are based on X = Final Payoff - Terminal Value, where:
- Final Payoff = max(S_T - K, 0) for a call option
- Terminal Value = p + sum(Delta_t * (S_{t+1} - S_t))

Replication Error Metrics:
- Marginal metrics: MSE over time of mean, 95th percentile (p95), 5th percentile (p05)
- Temporal dependencies: MSE between quadratic variations (QVar)
- Correlation structure: Time-averaged MSE between covariance matrices
"""

import torch
from typing import Dict, Any
from pathlib import Path
import json
from datetime import datetime

from src.hedging_models.base_hedger import DeepHedgingModel, NonDeepHedgingModel
from src.hedging_models.deep_hedgers.feedforward_layers import FeedforwardLayers
from src.hedging_models.deep_hedgers.feedforward_time import FeedforwardTime
from src.hedging_models.deep_hedgers.rnn_hedger import RNN
from src.hedging_models.deep_hedgers.lstm_hedger import LSTM

from src.hedging_models.non_deep_hedgers.black_scholes import BlackScholes
from src.hedging_models.non_deep_hedgers.delta_gamma import DeltaGamma
from src.hedging_models.non_deep_hedgers.linear_regression import LinearRegression
from src.hedging_models.non_deep_hedgers.xgboost import XGBoost

from src.utils.preprocessing_utils import LogReturnTransformation


def log_returns_to_prices(
    log_returns: torch.Tensor,
    initial_prices: torch.Tensor
) -> torch.Tensor:
    """
    Convert log returns to prices using initial prices.
    Assumes both inputs are torch tensors.
    """
    if log_returns.ndim != 2:
        raise ValueError(f"Expected 2D tensor (R, L), got {log_returns.ndim}D")
    
    R, L = log_returns.shape
    
    if initial_prices.shape != (R,):
        raise ValueError(f"initial_prices shape {initial_prices.shape} doesn't match expected (R,)")
    
    scaler = LogReturnTransformation()
    prices = torch.zeros((R, L), device=log_returns.device)
    
    for i in range(R):
        prices_full = scaler.inverse_transform(log_returns[i], initial_prices[i])
        prices[i] = prices_full[1:]
    
    return prices

def compute_replication_errors(hedger, prices: torch.Tensor) -> torch.Tensor:
    """
    Compute Replication Errors: R = Final Payoff - Terminal Value for each sample.
    Assumes prices is a torch tensor, returns torch tensor.
    """
    if isinstance(hedger, DeepHedgingModel):
        hedger.eval()
    prices = prices.to(hedger.device).float()
    with torch.no_grad():
        deltas = hedger.forward(prices)
        terminal_values = hedger.compute_terminal_value(prices, deltas)
        final_prices = prices[:, -1]
        payoffs = torch.clamp(final_prices - float(hedger.strike), min=0.0)  # European call option payoff
        R = payoffs - terminal_values
    return R


def compute_marginal_metrics(X_real: torch.Tensor, X_synthetic: torch.Tensor) -> Dict[str, float]:
    """
    Compute marginal distribution metrics: MSE over time of mean, p95, p05.
    Assumes both inputs are torch tensors.
    """
    mean_real = X_real.mean()
    mean_syn = X_synthetic.mean()
    p95_real = X_real.quantile(0.95)
    p95_syn = X_synthetic.quantile(0.95)
    p05_real = X_real.quantile(0.05)
    p05_syn = X_synthetic.quantile(0.05)
    mse_mean = float((mean_real - mean_syn).pow(2).item())
    mse_p95 = float((p95_real - p95_syn).pow(2).item())
    mse_p05 = float((p05_real - p05_syn).pow(2).item())
    return {
        'mse_mean': mse_mean,
        'mse_p95': mse_p95,
        'mse_p05': mse_p05,
        'mean_real': float(mean_real.item()),
        'mean_syn': float(mean_syn.item()),
        'p95_real': float(p95_real.item()),
        'p95_syn': float(p95_syn.item()),
        'p05_real': float(p05_real.item()),
        'p05_syn': float(p05_syn.item())
    }


def compute_quadratic_variation(prices: torch.Tensor) -> torch.Tensor:
    """
    Compute quadratic variation of price series.
    Assumes input is a torch tensor.
    """
    if prices.ndim == 2:
        # (R, L)
        price_diffs = torch.diff(prices, dim=1)  # (R, L-1)
        qvar = torch.sum(price_diffs ** 2, dim=1)  # (R,)
    elif prices.ndim == 3:
        # (R, L, N)
        price_diffs = torch.diff(prices, dim=1)  # (R, L-1, N)
        qvar = torch.sum(price_diffs ** 2, dim=1)  # (R, N)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {prices.ndim}D")
    return qvar


def compute_temporal_metrics(
    prices_real: torch.Tensor,
    prices_synthetic: torch.Tensor
) -> Dict[str, float]:
    """
    Compute temporal dependency metrics: MSE between quadratic variations.
    Assumes both inputs are torch tensors.
    """
    qvar_real = compute_quadratic_variation(prices_real)
    qvar_syn = compute_quadratic_variation(prices_synthetic)
    # Compute MSE for vectors
    mse_qvar = float(torch.mean((qvar_real - qvar_syn) ** 2).item())
    return {
        'mse_qvar': mse_qvar,
        'mean_qvar_real': float(qvar_real.mean().item()),
        'mean_qvar_syn': float(qvar_syn.mean().item())
    }


def compute_covariance_matrix(prices: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance matrix of price series.
    Assumes input is a torch tensor.
    """
    if prices.ndim == 2:
        # (R, L) - covariance across samples for each time step
        cov = torch.cov(prices.T)  # (L, L)
    elif prices.ndim == 3:
        # (R, L, N) - flatten to (R, L*N) then compute covariance
        R, L, N = prices.shape
        prices_flat = prices.reshape(R, L * N)
        cov = torch.cov(prices_flat.T)  # (L*N, L*N)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {prices.ndim}D")
    return cov


def compute_correlation_metrics(
    prices_real: torch.Tensor,
    prices_synthetic: torch.Tensor
) -> Dict[str, float]:
    """
    Compute correlation structure metrics: time-averaged MSE between covariance matrices.
    Assumes both inputs are torch tensors.
    """
    cov_real = compute_covariance_matrix(prices_real)
    cov_syn = compute_covariance_matrix(prices_synthetic)
    # MSE between covariance matrices
    mse_cov = float(torch.mean((cov_real - cov_syn) ** 2).item())
    return {
        'mse_cov': mse_cov,
        'mean_cov_real': float(cov_real.mean().item()),
        'mean_cov_syn': float(cov_syn.mean().item())
    }


def fit_hedger(
    hedger,
    data: torch.Tensor,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3
):
    """
    Fit a hedging model with appropriate arguments based on its type.
    
    Args:
        hedger: The hedging model instance (DeepHedgingModel or NonDeepHedgingModel)
        data: Training data tensor
        num_epochs: Number of training epochs (only for DeepHedgingModel)
        batch_size: Batch size (only for DeepHedgingModel)
        learning_rate: Learning rate (only for DeepHedgingModel)
    """
    if isinstance(hedger, DeepHedgingModel):
        hedger.fit(data, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
    elif isinstance(hedger, NonDeepHedgingModel):
        hedger.fit(data)
    else:
        raise ValueError(f"Unknown hedger type: {type(hedger)}")

class AugmentedTestingEvaluator:
    def __init__(
        self,
        real_train_log_returns: torch.Tensor,
        real_val_log_returns: torch.Tensor,
        synthetic_train_log_returns: torch.Tensor,
        real_train_initial: torch.Tensor,
        real_val_initial: torch.Tensor,
        synthetic_train_initial: torch.Tensor = None,
        seq_length: int = None,
        num_epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3
    ):
        print("[AugmentedTestingEvaluator] Initialization started...")
        
        self.real_train_log_returns = real_train_log_returns
        self.real_val_log_returns = real_val_log_returns
        self.synthetic_train_log_returns = synthetic_train_log_returns
        self.real_train_initial = real_train_initial
        self.real_val_initial = real_val_initial
        
        if seq_length is None:
            self.seq_length = self.real_train_log_returns.shape[1]
        else:
            self.seq_length = seq_length
            
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if synthetic_train_initial is None:
            R_syn = synthetic_train_log_returns.shape[0]
            mean_initial = float(self.real_train_initial.mean().item())
            self.synthetic_train_initial = torch.ones(R_syn, device=self.real_train_initial.device) * mean_initial
        else:
            self.synthetic_train_initial = synthetic_train_initial.float()
        
        # Convert log returns to prices (returns torch tensors)
        print("[AugmentedTestingEvaluator] Converting log returns to prices for real training data...")
        self.real_train_prices = log_returns_to_prices(self.real_train_log_returns, self.real_train_initial)
        
        print("[AugmentedTestingEvaluator] Converting log returns to prices for real validation data...")
        self.real_val_prices = log_returns_to_prices(self.real_val_log_returns, self.real_val_initial)
        
        print("[AugmentedTestingEvaluator] Converting log returns to prices for synthetic training data...")
        self.synthetic_train_prices = log_returns_to_prices(self.synthetic_train_log_returns, self.synthetic_train_initial)
        
        self.strike = float(self.real_train_initial.mean().item())
        print("[AugmentedTestingEvaluator] Initialization complete.")

    def evaluate(self, hedger_class) -> Dict[str, Any]:
        """
        Evaluate augmented testing for a given hedger class.
        """
        print("[AugmentedTestingEvaluator] Starting evaluation procedure...")
        # Mix synthetic and real data (50/50)
        R_real = self.real_train_prices.shape[0]
        R_syn = self.synthetic_train_prices.shape[0]
        R_mixed = min(R_real, R_syn)
        
        print(f"[AugmentedTestingEvaluator] Mixing {R_mixed} real and {R_mixed} synthetic training samples (50/50)...")
        torch.manual_seed(42)
        real_indices = torch.randperm(R_real)[:R_mixed]
        syn_indices = torch.randperm(R_syn)[:R_mixed]
        
        real_train_subset = self.real_train_prices[real_indices]
        synthetic_train_subset = self.synthetic_train_prices[syn_indices]
        
        mixed_train_prices = torch.cat([
            real_train_subset,
            synthetic_train_subset
        ], dim=0)
        shuffle_idx = torch.randperm(mixed_train_prices.shape[0])
        mixed_train_prices = mixed_train_prices[shuffle_idx]
        
        print("[AugmentedTestingEvaluator] Training hedger on mixed (synthetic + real) data...")
        hedger_mixed = hedger_class(
            seq_length=self.seq_length,
            strike=self.strike
        )
        fit_hedger(
            hedger_mixed,
            mixed_train_prices,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate
        )
        
        print("[AugmentedTestingEvaluator] Training hedger on real data only...")
        hedger_real = hedger_class(
            seq_length=self.seq_length,
            strike=self.strike
        )
        fit_hedger(
            hedger_real,
            self.real_train_prices,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate
        )
        
        print("[AugmentedTestingEvaluator] Evaluating both hedgers on real validation set...")
        X_mixed = compute_replication_errors(hedger_mixed, self.real_val_prices)
        X_real_only = compute_replication_errors(hedger_real, self.real_val_prices)
        
        print("[AugmentedTestingEvaluator] Computing metrics for mixed and real-only hedgers...")
        marginal_mixed = compute_marginal_metrics(X_real_only, X_mixed)
        temporal_mixed = compute_temporal_metrics(real_train_subset, synthetic_train_subset)
        correlation_mixed = compute_correlation_metrics(real_train_subset, synthetic_train_subset)
        
        results = {
            'mixed_training': {
                'marginal_metrics': marginal_mixed,
                'temporal_metrics': temporal_mixed,
                'correlation_metrics': correlation_mixed,
                'mean_X_mixed': float(X_mixed.mean().item()),
                'std_X_mixed': float(X_mixed.std().item()),
                'mse_X_mixed': float(torch.mean(X_mixed ** 2).item())
            },
            'real_only_training': {
                'mean_X_real': float(X_real_only.mean().item()),
                'std_X_real': float(X_real_only.std().item()),
                'mse_X_real': float(torch.mean(X_real_only ** 2).item())
            },
            'score': {
                'mse_mean': marginal_mixed['mse_mean'],
                'mse_p95': marginal_mixed['mse_p95'],
                'mse_p05': marginal_mixed['mse_p05'],
                'mse_qvar': temporal_mixed['mse_qvar'],
                'mse_cov': correlation_mixed['mse_cov']
            }
        }
        print("[AugmentedTestingEvaluator] Evaluation complete.")
        return results


class AlgorithmComparisonEvaluator:
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
        synthetic_train_initial: torch.Tensor = None,
        synthetic_val_initial: torch.Tensor = None,
        synthetic_test_initial: torch.Tensor = None,
        seq_length: int = None,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3
    ):
        print("[AlgorithmComparisonEvaluator] Initialization started...")
        
        self.real_train_log_returns = real_train_log_returns
        self.real_val_log_returns = real_val_log_returns
        self.real_test_log_returns = real_test_log_returns
        self.real_train_initial = real_train_initial
        self.real_val_initial = real_val_initial
        self.real_test_initial = real_test_initial
        
        if seq_length is None:
            self.seq_length = self.real_train_log_returns.shape[1]
        else:
            self.seq_length = seq_length
            
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        mean_initial = float(self.real_train_initial.mean().item())
        device = self.real_train_initial.device
        
        if synthetic_train_initial is None:
            R_syn = synthetic_train_log_returns.shape[0]
            self.synthetic_train_initial = torch.ones(R_syn, device=device) * mean_initial
        else:
            self.synthetic_train_initial = synthetic_train_initial.float()
            
        if synthetic_val_initial is None:
            R_syn = synthetic_val_log_returns.shape[0]
            self.synthetic_val_initial = torch.ones(R_syn, device=device) * mean_initial
        else:
            self.synthetic_val_initial = synthetic_val_initial.float()
            
        if synthetic_test_initial is None:
            R_syn = synthetic_test_log_returns.shape[0]
            self.synthetic_test_initial = torch.ones(R_syn, device=device) * mean_initial
        else:
            self.synthetic_test_initial = synthetic_test_initial.float()
        
        synthetic_train_log_returns = synthetic_train_log_returns.float()
        synthetic_val_log_returns = synthetic_val_log_returns.float()
        synthetic_test_log_returns = synthetic_test_log_returns.float()
        
        print("Converting log returns to prices for real data...")
        self.real_train_prices = log_returns_to_prices(self.real_train_log_returns, self.real_train_initial)
        self.real_val_prices = log_returns_to_prices(self.real_val_log_returns, self.real_val_initial)
        self.real_test_prices = log_returns_to_prices(self.real_test_log_returns, self.real_test_initial)
        
        print("[AlgorithmComparisonEvaluator] Converting log returns to prices for synthetic data...")
        self.synthetic_train_prices = log_returns_to_prices(synthetic_train_log_returns, self.synthetic_train_initial)
        self.synthetic_val_prices = log_returns_to_prices(synthetic_val_log_returns, self.synthetic_val_initial)
        self.synthetic_test_prices = log_returns_to_prices(synthetic_test_log_returns, self.synthetic_test_initial)
        
        self.strike = mean_initial

        self.hedger_classes = {
            'Feedforward_L-1': FeedforwardLayers,
            'Feedforward_Time': FeedforwardTime,
            'RNN': RNN,
            'LSTM': LSTM
        }

        self.hedger_classes.update({
            'BlackScholes': BlackScholes,
            'DeltaGamma': DeltaGamma,
            'LinearRegression': LinearRegression,
            'XGBoost': XGBoost,
        })
        print("[AlgorithmComparisonEvaluator] Initialization complete.")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate all four hedging models on both real and synthetic data.
        """
        print("[AlgorithmComparisonEvaluator] Starting evaluation procedure for all hedging models...")
        results = {}
        
        for hedger_name, hedger_class in self.hedger_classes.items():
            print(f"[AlgorithmComparisonEvaluator] --- Evaluating {hedger_name} ---")
            
            print(f"[AlgorithmComparisonEvaluator] Training {hedger_name} on real data...")
            hedger_real = hedger_class(
                seq_length=self.seq_length,
                strike=self.strike
            )
            fit_hedger(
                hedger_real,
                self.real_train_prices,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate
            )
            
            print(f"[AlgorithmComparisonEvaluator] Training {hedger_name} on synthetic data...")
            hedger_syn = hedger_class(
                seq_length=self.seq_length,
                strike=self.strike
            )
            fit_hedger(
                hedger_syn,
                self.synthetic_train_prices,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate
            )
            
            print(f"[AlgorithmComparisonEvaluator] Evaluating {hedger_name} on real test set...")
            X_real_test_real_hedger = compute_replication_errors(
                hedger_real,
                self.real_test_prices,
            )
            X_real_test_syn_hedger = compute_replication_errors(
                hedger_syn,
                self.real_test_prices,
            )
            
            print(f"[AlgorithmComparisonEvaluator] Evaluating {hedger_name} on synthetic test set...")
            X_syn_test_real_hedger = compute_replication_errors(
                hedger_real,
                self.synthetic_test_prices,
            )
            X_syn_test_syn_hedger = compute_replication_errors(
                hedger_syn,
                self.synthetic_test_prices,
            )
            
            print(f"[AlgorithmComparisonEvaluator] Computing metrics for {hedger_name} on real and synthetic test sets...")
            marginal_real = compute_marginal_metrics(X_real_test_real_hedger, X_real_test_syn_hedger)
            R_real_test = self.real_test_prices.shape[0]
            R_syn_test = self.synthetic_test_prices.shape[0]
            R_compare = min(R_real_test, R_syn_test)
            
            torch.manual_seed(42)
            real_test_indices = torch.randperm(R_real_test)[:R_compare]
            real_test_subset = self.real_test_prices[real_test_indices]
            synthetic_test_subset = self.synthetic_test_prices[:R_compare]
            
            temporal_real = compute_temporal_metrics(real_test_subset, synthetic_test_subset)
            correlation_real = compute_correlation_metrics(real_test_subset, synthetic_test_subset)
            
            marginal_syn = compute_marginal_metrics(X_syn_test_real_hedger, X_syn_test_syn_hedger)
            temporal_syn = compute_temporal_metrics(real_test_subset, synthetic_test_subset)
            correlation_syn = compute_correlation_metrics(real_test_subset, synthetic_test_subset)
            
            score_vector = [
                marginal_real['mse_mean'] + marginal_real['mse_p95'] + marginal_real['mse_p05'],  # Real test hedger comparison
                temporal_real['mse_qvar'],  # Temporal structure preservation
                correlation_real['mse_cov'],  # Correlation structure preservation
                marginal_syn['mse_mean'] + marginal_syn['mse_p95'] + marginal_syn['mse_p05']  # Synthetic test hedger comparison
            ]
            
            results[hedger_name] = {
                'real_test': {
                    'marginal_metrics': marginal_real,
                    'temporal_metrics': temporal_real,
                    'correlation_metrics': correlation_real,
                    'mean_X_real_hedger': float(X_real_test_real_hedger.mean().item()),
                    'std_X_real_hedger': float(X_real_test_real_hedger.std().item()),
                    'mse_X_real_hedger': float(torch.mean(X_real_test_real_hedger ** 2).item()),
                    'mean_X_syn_hedger': float(X_real_test_syn_hedger.mean().item()),
                    'std_X_syn_hedger': float(X_real_test_syn_hedger.std().item()),
                    'mse_X_syn_hedger': float(torch.mean(X_real_test_syn_hedger ** 2).item())
                },
                'synthetic_test': {
                    'marginal_metrics': marginal_syn,
                    'temporal_metrics': temporal_syn,
                    'correlation_metrics': correlation_syn,
                    'mean_X_real_hedger': float(X_syn_test_real_hedger.mean().item()),
                    'std_X_real_hedger': float(X_syn_test_real_hedger.std().item()),
                    'mse_X_real_hedger': float(torch.mean(X_syn_test_real_hedger ** 2).item()),
                    'mean_X_syn_hedger': float(X_syn_test_syn_hedger.mean().item()),
                    'std_X_syn_hedger': float(X_syn_test_syn_hedger.std().item()),
                    'mse_X_syn_hedger': float(torch.mean(X_syn_test_syn_hedger ** 2).item())
                },
                'score_vector': score_vector
            }
            print(f"[AlgorithmComparisonEvaluator] --- Done with {hedger_name} ---\n")
        
        print("[AlgorithmComparisonEvaluator] Evaluation of all hedgers complete.")
        return results


def save_utility_scores(results: Dict[str, Any], model_name: str, results_dir: Path = None):
    """
    Save utility scores to JSON file.
    """
    print("[save_utility_scores] Saving utility scores...")
    if results_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / "results" / "utility_scores"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    output = {
        'model_name': model_name,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'results': results
    }
    
    output_path = results_dir / f"{model_name}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"[save_utility_scores] Utility scores saved to {output_path}")
    return output_path

