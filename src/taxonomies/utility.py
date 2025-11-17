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
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import json
from datetime import datetime

from src.hedging_models.feedforward_layers import FeedforwardLayers
from src.hedging_models.feedforward_time import FeedforwardTime
from src.hedging_models.rnn_hedger import RNN
from src.hedging_models.lstm_hedger import LSTM
from src.hedging_models.non_deep_hedgers import (
    BlackScholes,
    DeltaGamma,
    RandomForest,
    LinearRegression,
    XGBoost,
    LightGBM
)

from src.utils.preprocessing_utils import LogReturnTransformation


def log_returns_to_prices(
    log_returns: torch.Tensor,
    initial_prices: torch.Tensor
) -> torch.Tensor:
    """
    Convert log returns to prices using initial prices.
    Assumes both inputs are torch tensors.
    """
    assert isinstance(log_returns, torch.Tensor), "log_returns must be a torch.Tensor"
    assert isinstance(initial_prices, torch.Tensor), "initial_prices must be a torch.Tensor"
    assert log_returns.ndim == 2, f"Expected 2D tensor (R, L), got {log_returns.ndim}D"
    
    R, L = log_returns.shape
    
    assert initial_prices.shape == (R,), f"initial_prices shape {initial_prices.shape} doesn't match expected (R,)"
    
    scaler = LogReturnTransformation()
    prices_np = np.zeros((R, L))
    
    # Convert to numpy for LogReturnTransformation (which works with numpy)
    log_returns_np = log_returns.cpu().numpy()
    initial_prices_np = initial_prices.cpu().numpy()
    
    for i in range(R):
        prices_full = scaler.inverse_transform(log_returns_np[i], initial_prices_np[i])
        prices_np[i] = prices_full[1:]
    
    return torch.from_numpy(prices_np).float()


def compute_replication_errors(hedger, prices: torch.Tensor, strike: float = 1.0) -> torch.Tensor:
    """
    Compute Replication Errors: R = Final Payoff - Terminal Value for each sample.
    Assumes prices is a torch tensor, returns torch tensor.
    """
    assert isinstance(prices, torch.Tensor), "prices must be a torch.Tensor"
    
    hedger.eval()
    
    prices = prices.to(hedger.device).float()
    
    with torch.no_grad():
        deltas = hedger.forward(prices)
        terminal_values = hedger.compute_terminal_value(prices, deltas)
        final_prices = prices[:, -1]
        payoffs = torch.clamp(final_prices - strike, min=0.0) # European call option payoff
        R = payoffs - terminal_values
    
    return R


def compute_marginal_metrics(X_real: torch.Tensor, X_synthetic: torch.Tensor) -> Dict[str, float]:
    """
    Compute marginal distribution metrics: MSE over time of mean, p95, p05.
    Assumes both inputs are torch tensors.
    """
    assert isinstance(X_real, torch.Tensor), "X_real must be a torch.Tensor"
    assert isinstance(X_synthetic, torch.Tensor), "X_synthetic must be a torch.Tensor"
    
    # Compute statistics
    mean_real = X_real.mean().item()
    mean_syn = X_synthetic.mean().item()
    
    # Convert to numpy for percentile calculation
    X_real_np = X_real.cpu().numpy()
    X_syn_np = X_synthetic.cpu().numpy()
    
    p95_real = np.percentile(X_real_np, 95)
    p95_syn = np.percentile(X_syn_np, 95)
    
    p05_real = np.percentile(X_real_np, 5)
    p05_syn = np.percentile(X_syn_np, 5)
    
    # MSE for each statistic
    mse_mean = (mean_real - mean_syn) ** 2
    mse_p95 = (p95_real - p95_syn) ** 2
    mse_p05 = (p05_real - p05_syn) ** 2
    
    return {
        'mse_mean': float(mse_mean),
        'mse_p95': float(mse_p95),
        'mse_p05': float(mse_p05),
        'mean_real': float(mean_real),
        'mean_syn': float(mean_syn),
        'p95_real': float(p95_real),
        'p95_syn': float(p95_syn),
        'p05_real': float(p05_real),
        'p05_syn': float(p05_syn)
    }


def compute_quadratic_variation(prices: torch.Tensor) -> torch.Tensor:
    """
    Compute quadratic variation of price series.
    Assumes input is a torch tensor.
    """
    assert isinstance(prices, torch.Tensor), "prices must be a torch.Tensor"
    
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
    assert isinstance(prices_real, torch.Tensor), "prices_real must be a torch.Tensor"
    assert isinstance(prices_synthetic, torch.Tensor), "prices_synthetic must be a torch.Tensor"
    
    qvar_real = compute_quadratic_variation(prices_real)
    qvar_syn = compute_quadratic_variation(prices_synthetic)
    
    # Compute MSE
    mse_qvar = torch.mean((qvar_real - qvar_syn) ** 2).item()
    
    return {
        'mse_qvar': float(mse_qvar),
        'mean_qvar_real': float(qvar_real.mean().item()),
        'mean_qvar_syn': float(qvar_syn.mean().item())
    }


def compute_covariance_matrix(prices: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance matrix of price series.
    Assumes input is a torch tensor.
    """
    assert isinstance(prices, torch.Tensor), "prices must be a torch.Tensor"
    
    if prices.ndim == 2:
        # (R, L) - covariance across samples for each time step
        # Convert to numpy for np.cov, then back to tensor
        prices_np = prices.cpu().numpy()
        cov_np = np.cov(prices_np.T)  # (L, L)
        cov = torch.from_numpy(cov_np).float()
    elif prices.ndim == 3:
        # (R, L, N) - flatten to (R, L*N) then compute covariance
        R, L, N = prices.shape
        prices_flat = prices.reshape(R, L * N)
        prices_np = prices_flat.cpu().numpy()
        cov_np = np.cov(prices_np.T)  # (L*N, L*N)
        cov = torch.from_numpy(cov_np).float()
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
    assert isinstance(prices_real, torch.Tensor), "prices_real must be a torch.Tensor"
    assert isinstance(prices_synthetic, torch.Tensor), "prices_synthetic must be a torch.Tensor"
    
    cov_real = compute_covariance_matrix(prices_real)
    cov_syn = compute_covariance_matrix(prices_synthetic)
    
    # MSE between covariance matrices
    mse_cov = torch.mean((cov_real - cov_syn) ** 2).item()
    
    return {
        'mse_cov': float(mse_cov),
        'mean_cov_real': float(cov_real.mean().item()),
        'mean_cov_syn': float(cov_syn.mean().item())
    }


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
        strike: float = None,
        hidden_size: int = 64,
        num_epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 0.001
    ):
        print("[AugmentedTestingEvaluator] Initialization started...")
        # Assume all inputs are torch tensors
        assert isinstance(real_train_log_returns, torch.Tensor), "real_train_log_returns must be a torch.Tensor"
        assert isinstance(real_val_log_returns, torch.Tensor), "real_val_log_returns must be a torch.Tensor"
        assert isinstance(synthetic_train_log_returns, torch.Tensor), "synthetic_train_log_returns must be a torch.Tensor"
        assert isinstance(real_train_initial, torch.Tensor), "real_train_initial must be a torch.Tensor"
        assert isinstance(real_val_initial, torch.Tensor), "real_val_initial must be a torch.Tensor"
        
        self.real_train_log_returns = real_train_log_returns.float()
        self.real_val_log_returns = real_val_log_returns.float()
        self.synthetic_train_log_returns = synthetic_train_log_returns.float()
        self.real_train_initial = real_train_initial.float()
        self.real_val_initial = real_val_initial.float()
        
        if seq_length is None:
            self.seq_length = self.real_train_log_returns.shape[1]
        else:
            self.seq_length = seq_length
            
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if synthetic_train_initial is None:
            R_syn = synthetic_train_log_returns.shape[0]
            mean_initial = float(self.real_train_initial.mean().item())
            self.synthetic_train_initial = torch.ones(R_syn) * mean_initial
        else:
            assert isinstance(synthetic_train_initial, torch.Tensor), "synthetic_train_initial must be a torch.Tensor"
            self.synthetic_train_initial = synthetic_train_initial.float()
        
        # Convert log returns to prices (returns torch tensors)
        print("[AugmentedTestingEvaluator] Converting log returns to prices for real training data...")
        self.real_train_prices = log_returns_to_prices(self.real_train_log_returns, self.real_train_initial)
        
        print("[AugmentedTestingEvaluator] Converting log returns to prices for real validation data...")
        self.real_val_prices = log_returns_to_prices(self.real_val_log_returns, self.real_val_initial)
        
        print("[AugmentedTestingEvaluator] Converting log returns to prices for synthetic training data...")
        self.synthetic_train_prices = log_returns_to_prices(self.synthetic_train_log_returns, self.synthetic_train_initial)
        
        # Set strike price
        if strike is None:
            self.strike = float(self.real_train_initial.mean().item())  # At-the-money option
        else:
            self.strike = strike
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
        # Sample equal amounts from real and synthetic
        torch.manual_seed(42)
        real_indices = torch.randperm(R_real)[:R_mixed]
        syn_indices = torch.randperm(R_syn)[:R_mixed]
        
        # Extract the real and synthetic portions for comparison
        real_train_subset = self.real_train_prices[real_indices]
        synthetic_train_subset = self.synthetic_train_prices[syn_indices]
        
        mixed_train_prices = torch.cat([
            real_train_subset,
            synthetic_train_subset
        ], dim=0)
        # Shuffle
        shuffle_idx = torch.randperm(mixed_train_prices.shape[0])
        mixed_train_prices = mixed_train_prices[shuffle_idx]
        
        print("[AugmentedTestingEvaluator] Training hedger on mixed (synthetic + real) data...")
        # Train hedger on mixed data
        hedger_mixed = hedger_class(
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            strike=self.strike
        )
        hedger_mixed.fit(
            mixed_train_prices,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=True
        )
        
        print("[AugmentedTestingEvaluator] Training hedger on real data only...")
        # Train hedger on real data only
        hedger_real = hedger_class(
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            strike=self.strike
        )
        hedger_real.fit(
            self.real_train_prices,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=True
        )
        
        print("[AugmentedTestingEvaluator] Evaluating both hedgers on real validation set...")
        X_mixed = compute_replication_errors(hedger_mixed, self.real_val_prices, self.strike)
        X_real_only = compute_replication_errors(hedger_real, self.real_val_prices, self.strike)
        
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
        strike: float = None,
        hidden_size: int = 64,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        print("[AlgorithmComparisonEvaluator] Initialization started...")
        # Assume all inputs are torch tensors
        assert isinstance(real_train_log_returns, torch.Tensor), "real_train_log_returns must be a torch.Tensor"
        assert isinstance(real_val_log_returns, torch.Tensor), "real_val_log_returns must be a torch.Tensor"
        assert isinstance(real_test_log_returns, torch.Tensor), "real_test_log_returns must be a torch.Tensor"
        assert isinstance(synthetic_train_log_returns, torch.Tensor), "synthetic_train_log_returns must be a torch.Tensor"
        assert isinstance(synthetic_val_log_returns, torch.Tensor), "synthetic_val_log_returns must be a torch.Tensor"
        assert isinstance(synthetic_test_log_returns, torch.Tensor), "synthetic_test_log_returns must be a torch.Tensor"
        assert isinstance(real_train_initial, torch.Tensor), "real_train_initial must be a torch.Tensor"
        assert isinstance(real_val_initial, torch.Tensor), "real_val_initial must be a torch.Tensor"
        assert isinstance(real_test_initial, torch.Tensor), "real_test_initial must be a torch.Tensor"
        
        self.real_train_log_returns = real_train_log_returns.float()
        self.real_val_log_returns = real_val_log_returns.float()
        self.real_test_log_returns = real_test_log_returns.float()
        self.real_train_initial = real_train_initial.float()
        self.real_val_initial = real_val_initial.float()
        self.real_test_initial = real_test_initial.float()
        
        if seq_length is None:
            self.seq_length = self.real_train_log_returns.shape[1]
        else:
            self.seq_length = seq_length
            
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Handle synthetic initial prices
        mean_initial = float(self.real_train_initial.mean().item())
        if synthetic_train_initial is None:
            R_syn = synthetic_train_log_returns.shape[0]
            self.synthetic_train_initial = torch.ones(R_syn) * mean_initial
        else:
            assert isinstance(synthetic_train_initial, torch.Tensor), "synthetic_train_initial must be a torch.Tensor"
            self.synthetic_train_initial = synthetic_train_initial.float()
            
        if synthetic_val_initial is None:
            R_syn = synthetic_val_log_returns.shape[0]
            self.synthetic_val_initial = torch.ones(R_syn) * mean_initial
        else:
            assert isinstance(synthetic_val_initial, torch.Tensor), "synthetic_val_initial must be a torch.Tensor"
            self.synthetic_val_initial = synthetic_val_initial.float()
            
        if synthetic_test_initial is None:
            R_syn = synthetic_test_log_returns.shape[0]
            self.synthetic_test_initial = torch.ones(R_syn) * mean_initial
        else:
            assert isinstance(synthetic_test_initial, torch.Tensor), "synthetic_test_initial must be a torch.Tensor"
            self.synthetic_test_initial = synthetic_test_initial.float()
        
        # All synthetic log returns are already tensors
        synthetic_train_log_returns = synthetic_train_log_returns.float()
        synthetic_val_log_returns = synthetic_val_log_returns.float()
        synthetic_test_log_returns = synthetic_test_log_returns.float()
        
        # Convert log returns to prices (returns torch tensors)
        print("Converting log returns to prices for real data...")
        self.real_train_prices = log_returns_to_prices(self.real_train_log_returns, self.real_train_initial)
        self.real_val_prices = log_returns_to_prices(self.real_val_log_returns, self.real_val_initial)
        self.real_test_prices = log_returns_to_prices(self.real_test_log_returns, self.real_test_initial)
        
        print("[AlgorithmComparisonEvaluator] Converting log returns to prices for synthetic data...")
        self.synthetic_train_prices = log_returns_to_prices(synthetic_train_log_returns, self.synthetic_train_initial)
        self.synthetic_val_prices = log_returns_to_prices(synthetic_val_log_returns, self.synthetic_val_initial)
        self.synthetic_test_prices = log_returns_to_prices(synthetic_test_log_returns, self.synthetic_test_initial)
        
        # Set strike price
        if strike is None:
            self.strike = mean_initial  # At-the-money option
        else:
            self.strike = strike
        
        # Define all hedging model classes
        self.hedger_classes = {
            'Feedforward_L-1': FeedforwardLayers,
            'Feedforward_Time': FeedforwardTime,
            'RNN': RNN,
            'LSTM': LSTM
        }

        # Define all non-deep hedging model classes
        self.hedger_classes.update({
            'BlackScholes': BlackScholes,
            'DeltaGamma': DeltaGamma,
            'RandomForest': RandomForest,
            'LinearRegression': LinearRegression,
            'XGBoost': XGBoost,
            'LightGBM': LightGBM
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
            # Train on real data
            hedger_real = hedger_class(
                seq_length=self.seq_length,
                hidden_size=self.hidden_size,
                strike=self.strike
            )
            hedger_real.fit(
                self.real_train_prices,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                verbose=True
            )
            
            print(f"[AlgorithmComparisonEvaluator] Training {hedger_name} on synthetic data...")
            # Train on synthetic data
            hedger_syn = hedger_class(
                seq_length=self.seq_length,
                hidden_size=self.hidden_size,
                strike=self.strike
            )
            hedger_syn.fit(
                self.synthetic_train_prices,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                verbose=True
            )
            
            print(f"[AlgorithmComparisonEvaluator] Evaluating {hedger_name} on real test set...")
            # Evaluate both hedgers on real test set
            X_real_test_real_hedger = compute_replication_errors(
                hedger_real,
                self.real_test_prices,
                self.strike
            )
            X_real_test_syn_hedger = compute_replication_errors(
                hedger_syn,
                self.real_test_prices,
                self.strike
            )
            
            print(f"[AlgorithmComparisonEvaluator] Evaluating {hedger_name} on synthetic test set...")
            # Evaluate both hedgers on synthetic test set
            X_syn_test_real_hedger = compute_replication_errors(
                hedger_real,
                self.synthetic_test_prices,
                self.strike
            )
            X_syn_test_syn_hedger = compute_replication_errors(
                hedger_syn,
                self.synthetic_test_prices,
                self.strike
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

