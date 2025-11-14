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

from src.deep_hedgers.feedforward_layers import FeedforwardDeepHedger
from src.deep_hedgers.feedforward_time import FeedforwardTimeDeepHedger
from src.deep_hedgers.rnn_hedger import RNNDeepHedger
from src.deep_hedgers.lstm_hedger import LSTMDeepHedger
from src.utils.preprocessing_utils import LogReturnTransformation


def log_returns_to_prices(
    log_returns,
    initial_prices
):
    """
    Convert log returns to prices using initial prices.
    
    Args:
        log_returns: Tensor or array of shape (R, L) of log returns (univariate)
        initial_prices: Tensor or array of initial prices, shape (R,)
    
    Returns:
        prices: Tensor of shape (R, L) with prices (same type as input)
    """
    is_tensor = isinstance(log_returns, torch.Tensor)
    
    if is_tensor:
        log_returns_np = log_returns.cpu().numpy()
        initial_prices_np = initial_prices.cpu().numpy() if isinstance(initial_prices, torch.Tensor) else np.asarray(initial_prices)
    else:
        log_returns_np = np.asarray(log_returns)
        initial_prices_np = np.asarray(initial_prices)
    
    if log_returns_np.ndim != 2:
        raise ValueError(f"Expected 2D array (R, L), got {log_returns_np.ndim}D")
    
    R, L = log_returns_np.shape
    
    if initial_prices_np.shape != (R,):
        raise ValueError(f"initial_prices shape {initial_prices_np.shape} doesn't match expected (R,)")
    
    scaler = LogReturnTransformation()
    prices_np = np.zeros((R, L))
    
    for i in range(R):
        prices_full = scaler.inverse_transform(log_returns_np[i], initial_prices_np[i])
        prices_np[i] = prices_full[1:]  # Remove initial price to match log returns length
    
    if is_tensor:
        return torch.from_numpy(prices_np).float()
    return prices_np


def compute_X_values(hedger, prices: torch.Tensor, strike: float = 1.0) -> np.ndarray:
    """
    Compute X = Final Payoff - Terminal Value for each sample.
    
    Args:
        hedger: Trained deep hedging model
        prices: Price sequences of shape (R, L) - open channel only, must be torch.Tensor
        strike: Strike price for the call option
    
    Returns:
        X values of shape (R,)
    """
    hedger.eval()
    
    # Ensure prices is a torch tensor
    if not isinstance(prices, torch.Tensor):
        raise TypeError(f"prices must be torch.Tensor, got {type(prices)}")
    
    prices = prices.to(hedger.device).float()
    
    with torch.no_grad():
        deltas = hedger.forward(prices)
        terminal_values = hedger.compute_terminal_value(prices, deltas)
        final_prices = prices[:, -1]
        payoffs = torch.clamp(final_prices - strike, min=0.0)
        X = payoffs - terminal_values
    
    return X.cpu().numpy()


def compute_marginal_metrics(X_real: np.ndarray, X_synthetic: np.ndarray) -> Dict[str, float]:
    """
    Compute marginal distribution metrics: MSE over time of mean, p95, p05.
    
    Args:
        X_real: X values from real data, shape (R,)
        X_synthetic: X values from synthetic data, shape (R,)
    
    Returns:
        Dictionary with marginal metric scores
    """
    # Compute statistics
    mean_real = np.mean(X_real)
    mean_syn = np.mean(X_synthetic)
    
    p95_real = np.percentile(X_real, 95)
    p95_syn = np.percentile(X_synthetic, 95)
    
    p05_real = np.percentile(X_real, 5)
    p05_syn = np.percentile(X_synthetic, 5)
    
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


def compute_quadratic_variation(prices) -> np.ndarray:
    """
    Compute quadratic variation of price series.
    
    QVar = sum((S_{t+1} - S_t)^2) over time
    
    Args:
        prices: Price sequences of shape (R, L) or (R, L, N) (tensor or array)
    
    Returns:
        Quadratic variation of shape (R,) or (R, N) as numpy array
    """
    if isinstance(prices, torch.Tensor):
        prices = prices.cpu().numpy()
    else:
        prices = np.asarray(prices)
    
    if prices.ndim == 2:
        # (R, L)
        price_diffs = np.diff(prices, axis=1)  # (R, L-1)
        qvar = np.sum(price_diffs ** 2, axis=1)  # (R,)
    elif prices.ndim == 3:
        # (R, L, N)
        price_diffs = np.diff(prices, axis=1)  # (R, L-1, N)
        qvar = np.sum(price_diffs ** 2, axis=1)  # (R, N)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {prices.ndim}D")
    
    return qvar


def compute_temporal_metrics(
    prices_real, 
    prices_synthetic
) -> Dict[str, float]:
    """
    Compute temporal dependency metrics: MSE between quadratic variations.
    
    Args:
        prices_real: Real price sequences, shape (R, L) or (R, L, N) (tensor or array)
        prices_synthetic: Synthetic price sequences, shape (R, L) or (R, L, N) (tensor or array)
    
    Returns:
        Dictionary with temporal metric scores
    """
    qvar_real = compute_quadratic_variation(prices_real)
    qvar_syn = compute_quadratic_variation(prices_synthetic)
    
    if qvar_real.ndim == 1:
        # (R,)
        mse_qvar = np.mean((qvar_real - qvar_syn) ** 2)
    else:
        # (R, N) - average over channels
        mse_qvar = np.mean((qvar_real - qvar_syn) ** 2)
    
    return {
        'mse_qvar': float(mse_qvar),
        'mean_qvar_real': float(np.mean(qvar_real)),
        'mean_qvar_syn': float(np.mean(qvar_syn))
    }


def compute_covariance_matrix(prices) -> np.ndarray:
    """
    Compute covariance matrix of price series.
    
    Args:
        prices: Price sequences of shape (R, L) or (R, L, N) (tensor or array)
    
    Returns:
        Covariance matrix of shape (L, L) or (N, N) depending on input as numpy array
    """
    if isinstance(prices, torch.Tensor):
        prices = prices.cpu().numpy()
    else:
        prices = np.asarray(prices)
    
    if prices.ndim == 2:
        # (R, L) - covariance across samples for each time step
        # Transpose to get (L, R) then compute covariance
        cov = np.cov(prices.T)  # (L, L)
    elif prices.ndim == 3:
        # (R, L, N) - flatten to (R, L*N) then compute covariance
        R, L, N = prices.shape
        prices_flat = prices.reshape(R, L * N)
        cov = np.cov(prices_flat.T)  # (L*N, L*N)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {prices.ndim}D")
    
    return cov


def compute_correlation_metrics(
    prices_real,
    prices_synthetic
) -> Dict[str, float]:
    """
    Compute correlation structure metrics: time-averaged MSE between covariance matrices.
    
    Args:
        prices_real: Real price sequences, shape (R, L) or (R, L, N) (tensor or array)
        prices_synthetic: Synthetic price sequences, shape (R, L) or (R, L, N) (tensor or array)
    
    Returns:
        Dictionary with correlation metric scores
    """
    cov_real = compute_covariance_matrix(prices_real)
    cov_syn = compute_covariance_matrix(prices_synthetic)
    
    # MSE between covariance matrices
    mse_cov = np.mean((cov_real - cov_syn) ** 2)
    
    return {
        'mse_cov': float(mse_cov),
        'mean_cov_real': float(np.mean(cov_real)),
        'mean_cov_syn': float(np.mean(cov_syn))
    }


class AugmentedTestingEvaluator:
    """
    Augmented Testing: Mix synthetic with real training data (50/50), train hedger, compare with real-only.
    """
    
    def __init__(
        self,
        real_train_log_returns,
        real_val_log_returns,
        synthetic_train_log_returns,
        real_train_initial,
        real_val_initial,
        synthetic_train_initial = None,
        seq_length: int = None,
        strike: float = None,
        hidden_size: int = 64,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Args:
            real_train_log_returns: Real training log returns, shape (R_real, L) (tensor or array)
            real_val_log_returns: Real validation log returns, shape (R_val, L) (tensor or array)
            synthetic_train_log_returns: Synthetic training log returns, shape (R_syn, L) (tensor or array)
            real_train_initial: Real training initial prices, shape (R_real,) (tensor or array)
            real_val_initial: Real validation initial prices, shape (R_val,) (tensor or array)
            synthetic_train_initial: Synthetic training initial prices, shape (R_syn,) (optional, uses mean of real if None)
            seq_length: Sequence length L (inferred from data if None)
            strike: Strike price for call option (if None, will use mean of real_train_initial)
            hidden_size: Hidden size for hedgers
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
        """
        print("[AugmentedTestingEvaluator] Initialization started...")
        # Keep as tensors if they are tensors
        if isinstance(real_train_log_returns, torch.Tensor):
            self.real_train_log_returns = real_train_log_returns
            self.real_val_log_returns = real_val_log_returns
            self.synthetic_train_log_returns = synthetic_train_log_returns
            self.real_train_initial = real_train_initial if isinstance(real_train_initial, torch.Tensor) else torch.from_numpy(np.asarray(real_train_initial)).float()
            self.real_val_initial = real_val_initial if isinstance(real_val_initial, torch.Tensor) else torch.from_numpy(np.asarray(real_val_initial)).float()
        else:
            self.real_train_log_returns = torch.from_numpy(np.asarray(real_train_log_returns)).float()
            self.real_val_log_returns = torch.from_numpy(np.asarray(real_val_log_returns)).float()
            self.synthetic_train_log_returns = torch.from_numpy(np.asarray(synthetic_train_log_returns)).float()
            self.real_train_initial = torch.from_numpy(np.asarray(real_train_initial)).float()
            self.real_val_initial = torch.from_numpy(np.asarray(real_val_initial)).float()
        
        if seq_length is None:
            self.seq_length = self.real_train_log_returns.shape[1]
        else:
            self.seq_length = seq_length
            
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Handle synthetic initial prices
        if synthetic_train_initial is None:
            mean_initial = float(self.real_train_initial.mean().item())
            R_syn = self.synthetic_train_log_returns.shape[0]
            self.synthetic_train_initial = torch.ones(R_syn) * mean_initial
        else:
            if isinstance(synthetic_train_initial, torch.Tensor):
                self.synthetic_train_initial = synthetic_train_initial
            else:
                self.synthetic_train_initial = torch.from_numpy(np.asarray(synthetic_train_initial)).float()
        
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
        
        Args:
            hedger_class: Class of the hedger (e.g., FeedforwardDeepHedger)
        
        Returns:
            Dictionary with evaluation results
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
        
        mixed_train_prices = torch.cat([
            self.real_train_prices[real_indices],
            self.synthetic_train_prices[syn_indices]
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
        # Evaluate on real validation set
        X_mixed = compute_X_values(hedger_mixed, self.real_val_prices, self.strike)
        X_real_only = compute_X_values(hedger_real, self.real_val_prices, self.strike)
        
        print("[AugmentedTestingEvaluator] Computing metrics for mixed and real-only hedgers...")
        # Compute metrics (convert to numpy only here)
        marginal_mixed = compute_marginal_metrics(X_real_only, X_mixed)
        temporal_mixed = compute_temporal_metrics(self.real_val_prices, self.real_val_prices)
        correlation_mixed = compute_correlation_metrics(self.real_val_prices, self.real_val_prices)
        
        results = {
            'mixed_training': {
                'marginal_metrics': marginal_mixed,
                'temporal_metrics': temporal_mixed,
                'correlation_metrics': correlation_mixed,
                'mean_X_mixed': float(np.mean(X_mixed)),
                'std_X_mixed': float(np.std(X_mixed)),
                'mse_X_mixed': float(np.mean(X_mixed ** 2))
            },
            'real_only_training': {
                'mean_X_real': float(np.mean(X_real_only)),
                'std_X_real': float(np.std(X_real_only)),
                'mse_X_real': float(np.mean(X_real_only ** 2))
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
    """
    Algorithm Comparison: Generate train/val/test synthetic data, train 4 hedgers on both real and synthetic, evaluate.
    """
    
    def __init__(
        self,
        real_train_log_returns,
        real_val_log_returns,
        real_test_log_returns,
        synthetic_train_log_returns,
        synthetic_val_log_returns,
        synthetic_test_log_returns,
        real_train_initial,
        real_val_initial,
        real_test_initial,
        synthetic_train_initial = None,
        synthetic_val_initial = None,
        synthetic_test_initial = None,
        seq_length: int = None,
        strike: float = None,
        hidden_size: int = 64,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Args:
            real_train_log_returns: Real training log returns, shape (R, L) (tensor or array)
            real_val_log_returns: Real validation log returns, shape (R, L) (tensor or array)
            real_test_log_returns: Real test log returns, shape (R, L) (tensor or array)
            synthetic_train_log_returns: Synthetic training log returns, shape (R, L) (tensor or array)
            synthetic_val_log_returns: Synthetic validation log returns, shape (R, L) (tensor or array)
            synthetic_test_log_returns: Synthetic test log returns, shape (R, L) (tensor or array)
            real_train_initial: Real training initial prices, shape (R,) (tensor or array)
            real_val_initial: Real validation initial prices, shape (R,) (tensor or array)
            real_test_initial: Real test initial prices, shape (R,) (tensor or array)
            synthetic_train_initial: Synthetic training initial prices, shape (R,) (optional, uses mean of real if None)
            synthetic_val_initial: Synthetic validation initial prices, shape (R,) (optional, uses mean of real if None)
            synthetic_test_initial: Synthetic test initial prices, shape (R,) (optional, uses mean of real if None)
            seq_length: Sequence length L (inferred from data if None)
            strike: Strike price for call option (if None, will use mean of real_train_initial)
            hidden_size: Hidden size for hedgers
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
        """
        print("[AlgorithmComparisonEvaluator] Initialization started...")
        # Keep as tensors if they are tensors
        if isinstance(real_train_log_returns, torch.Tensor):
            self.real_train_log_returns = real_train_log_returns
            self.real_val_log_returns = real_val_log_returns
            self.real_test_log_returns = real_test_log_returns
            self.real_train_initial = real_train_initial if isinstance(real_train_initial, torch.Tensor) else torch.from_numpy(np.asarray(real_train_initial)).float()
            self.real_val_initial = real_val_initial if isinstance(real_val_initial, torch.Tensor) else torch.from_numpy(np.asarray(real_val_initial)).float()
            self.real_test_initial = real_test_initial if isinstance(real_test_initial, torch.Tensor) else torch.from_numpy(np.asarray(real_test_initial)).float()
        else:
            self.real_train_log_returns = torch.from_numpy(np.asarray(real_train_log_returns)).float()
            self.real_val_log_returns = torch.from_numpy(np.asarray(real_val_log_returns)).float()
            self.real_test_log_returns = torch.from_numpy(np.asarray(real_test_log_returns)).float()
            self.real_train_initial = torch.from_numpy(np.asarray(real_train_initial)).float()
            self.real_val_initial = torch.from_numpy(np.asarray(real_val_initial)).float()
            self.real_test_initial = torch.from_numpy(np.asarray(real_test_initial)).float()
        
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
            R_syn = synthetic_train_log_returns.shape[0] if isinstance(synthetic_train_log_returns, torch.Tensor) else len(synthetic_train_log_returns)
            self.synthetic_train_initial = torch.ones(R_syn) * mean_initial
        else:
            self.synthetic_train_initial = synthetic_train_initial if isinstance(synthetic_train_initial, torch.Tensor) else torch.from_numpy(np.asarray(synthetic_train_initial)).float()
            
        if synthetic_val_initial is None:
            R_syn = synthetic_val_log_returns.shape[0] if isinstance(synthetic_val_log_returns, torch.Tensor) else len(synthetic_val_log_returns)
            self.synthetic_val_initial = torch.ones(R_syn) * mean_initial
        else:
            self.synthetic_val_initial = synthetic_val_initial if isinstance(synthetic_val_initial, torch.Tensor) else torch.from_numpy(np.asarray(synthetic_val_initial)).float()
            
        if synthetic_test_initial is None:
            R_syn = synthetic_test_log_returns.shape[0] if isinstance(synthetic_test_log_returns, torch.Tensor) else len(synthetic_test_log_returns)
            self.synthetic_test_initial = torch.ones(R_syn) * mean_initial
        else:
            self.synthetic_test_initial = synthetic_test_initial if isinstance(synthetic_test_initial, torch.Tensor) else torch.from_numpy(np.asarray(synthetic_test_initial)).float()
        
        # Convert synthetic log returns to tensors if needed
        if not isinstance(synthetic_train_log_returns, torch.Tensor):
            synthetic_train_log_returns = torch.from_numpy(np.asarray(synthetic_train_log_returns)).float()
        if not isinstance(synthetic_val_log_returns, torch.Tensor):
            synthetic_val_log_returns = torch.from_numpy(np.asarray(synthetic_val_log_returns)).float()
        if not isinstance(synthetic_test_log_returns, torch.Tensor):
            synthetic_test_log_returns = torch.from_numpy(np.asarray(synthetic_test_log_returns)).float()
        
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
        
        # Define all hedger classes
        self.hedger_classes = {
            'Feedforward_L-1': FeedforwardDeepHedger,
            'Feedforward_Time': FeedforwardTimeDeepHedger,
            'RNN': RNNDeepHedger,
            'LSTM': LSTMDeepHedger
        }
        print("[AlgorithmComparisonEvaluator] Initialization complete.")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate all four hedgers on both real and synthetic data.
        
        Returns:
            Dictionary with evaluation results for each hedger
        """
        print("[AlgorithmComparisonEvaluator] Starting evaluation procedure for all hedgers...")
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
            # Evaluate on real test set
            X_real_test = compute_X_values(
                hedger_real,
                self.real_test_prices,
                self.strike
            )
            
            print(f"[AlgorithmComparisonEvaluator] Evaluating {hedger_name} on synthetic test set...")
            # Evaluate on synthetic test set
            X_syn_test = compute_X_values(
                hedger_syn,
                self.synthetic_test_prices,
                self.strike
            )
            
            print(f"[AlgorithmComparisonEvaluator] Computing metrics for {hedger_name} on real and synthetic test sets...")
            # Compute metrics for real test set (convert to numpy only here)
            marginal_real = compute_marginal_metrics(X_real_test, X_real_test)  # Baseline
            temporal_real = compute_temporal_metrics(self.real_test_prices, self.real_test_prices)
            correlation_real = compute_correlation_metrics(self.real_test_prices, self.real_test_prices)
            
            # Compute metrics for synthetic test set (convert to numpy only here)
            marginal_syn = compute_marginal_metrics(X_syn_test, X_syn_test)  # Baseline
            temporal_syn = compute_temporal_metrics(self.synthetic_test_prices, self.synthetic_test_prices)
            correlation_syn = compute_correlation_metrics(self.synthetic_test_prices, self.synthetic_test_prices)
            
            # Compute score vector (4-dimensional)
            score_vector = [
                marginal_real['mse_mean'] + marginal_real['mse_p95'] + marginal_real['mse_p05'],
                temporal_real['mse_qvar'],
                correlation_real['mse_cov'],
                marginal_syn['mse_mean'] + marginal_syn['mse_p95'] + marginal_syn['mse_p05'] + 
                temporal_syn['mse_qvar'] + correlation_syn['mse_cov']
            ]
            
            results[hedger_name] = {
                'real_test': {
                    'marginal_metrics': marginal_real,
                    'temporal_metrics': temporal_real,
                    'correlation_metrics': correlation_real,
                    'mean_X': float(np.mean(X_real_test)),
                    'std_X': float(np.std(X_real_test)),
                    'mse_X': float(np.mean(X_real_test ** 2))
                },
                'synthetic_test': {
                    'marginal_metrics': marginal_syn,
                    'temporal_metrics': temporal_syn,
                    'correlation_metrics': correlation_syn,
                    'mean_X': float(np.mean(X_syn_test)),
                    'std_X': float(np.std(X_syn_test)),
                    'mse_X': float(np.mean(X_syn_test ** 2))
                },
                'score_vector': score_vector
            }
            print(f"[AlgorithmComparisonEvaluator] --- Done with {hedger_name} ---\n")
        
        print("[AlgorithmComparisonEvaluator] Evaluation of all hedgers complete.")
        return results


def save_utility_scores(results: Dict[str, Any], model_name: str, results_dir: Path = None):
    """
    Save utility scores to JSON file.
    
    Args:
        results: Dictionary containing evaluation results
        model_name: Name of the generative model
        results_dir: Directory to save results (default: results/utility_scores/)
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

