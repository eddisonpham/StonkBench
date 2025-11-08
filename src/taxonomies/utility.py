"""
Utility-based evaluation for deep hedging models.

This module implements two types of evaluations:
1. Augmented Testing: Mix synthetic with real training data (50/50), train hedger, compare with real-only
2. Algorithm Comparison: Generate train/val/test synthetic data, train 4 hedgers on both real and synthetic, evaluate

The scores are based on X = Final Payoff - Terminal Value, where:
- Final Payoff = max(S_T - K, 0) for a call option
- Terminal Value = p + sum(Delta_t * (S_{t+1} - S_t))

Metrics:
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


def log_returns_to_prices(
    log_returns: np.ndarray, 
    initial_prices: np.ndarray = None,
    initial_price: float = None
) -> np.ndarray:
    """
    Convert log returns to prices.
    
    Args:
        log_returns: Array of shape (R, L, N) or (L, N) of log returns
        initial_prices: Initial prices for each sample, shape (R, N) for 3D log_returns or (N,) for 2D
                        If None, uses initial_price for all samples
        initial_price: Single initial price value to use if initial_prices is None (default: 1.0)
    
    Returns:
        prices: Array of same shape as log_returns but with prices
    """
    log_returns_np = np.asarray(log_returns)
    
    if initial_price is None:
        initial_price = 1.0
    
    if log_returns_np.ndim == 2:
        L, N = log_returns_np.shape
        prices = np.zeros((L + 1, N))
        
        if initial_prices is not None:
            initial_prices = np.asarray(initial_prices)
            if initial_prices.shape == (N,):
                prices[0] = initial_prices
            else:
                raise ValueError(f"initial_prices shape {initial_prices.shape} doesn't match expected (N,)")
        else:
            prices[0] = initial_price
        
        for t in range(L):
            prices[t + 1] = prices[t] * np.exp(log_returns_np[t])
        prices = prices[1:]
        
    elif log_returns_np.ndim == 3:
        R, L, N = log_returns_np.shape
        prices = np.zeros((R, L + 1, N))
        
        if initial_prices is not None:
            initial_prices = np.asarray(initial_prices)
            if initial_prices.shape == (R, N):
                prices[:, 0] = initial_prices
            elif initial_prices.shape == (N,):
                # Broadcast single initial price to all samples
                prices[:, 0] = initial_prices
            else:
                raise ValueError(f"initial_prices shape {initial_prices.shape} doesn't match expected (R, N) or (N,)")
        else:
            prices[:, 0] = initial_price
        
        for t in range(L):
            prices[:, t + 1] = prices[:, t] * np.exp(log_returns_np[:, t])
        prices = prices[:, 1:]
    else:
        raise ValueError(f"Expected 2D or 3D array, got {log_returns_np.ndim}D")
    
    return prices


def get_initial_prices_from_original_data(
    original_data_path: str,
    log_returns_windows: np.ndarray,
    window_size: int,
    channel_idx: int = 0
) -> np.ndarray:
    """
    Extract initial prices for each window from original price data.
    
    The windows are created from log returns, which are computed as:
    log_return[t] = log(price[t+1] / price[t])
    
    So a window of log returns [r_0, r_1, ..., r_{L-1}] corresponds to
    price transitions from price[0] to price[L], meaning the initial price
    for this window is price[0].
    
    Args:
        original_data_path: Path to original CSV file with price data
        log_returns_windows: Log returns windows of shape (R, L, N)
        window_size: Size of each window (L) - number of log returns
        channel_idx: Index of channel to extract (0 for Open)
    
    Returns:
        Initial prices for each window, shape (R,)
    """
    import pandas as pd
    
    # Read original price data
    df = pd.read_csv(original_data_path)
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close']
    original_prices = df[REQUIRED_COLUMNS].values  # (T, N)
    
    # Extract the channel we need (Open channel)
    prices_channel = original_prices[:, channel_idx]  # (T,)
    
    # Compute log returns from original prices to understand the mapping
    # log_returns[i] = log(prices[i+1] / prices[i])
    # So log_returns has length T-1
    
    # Windows are created with step=1 (overlapping windows)
    # Window 0: log_returns[0:L] corresponds to prices[0:L+1], initial = prices[0]
    # Window 1: log_returns[1:L+1] corresponds to prices[1:L+2], initial = prices[1]
    # Window i: log_returns[i:i+L] corresponds to prices[i:i+L+1], initial = prices[i]
    
    R = log_returns_windows.shape[0]
    initial_prices = np.zeros(R)
    
    # For each window i, the initial price is at index i in the original prices
    # But we need to make sure we don't go out of bounds
    max_start_idx = len(prices_channel) - window_size - 1
    
    for i in range(R):
        if i <= max_start_idx:
            initial_prices[i] = prices_channel[i]
        else:
            # If we run out of data, use the last available initial price
            # or repeat the last valid window's initial price
            initial_prices[i] = prices_channel[min(i, max_start_idx)] if max_start_idx >= 0 else prices_channel[0]
    
    return initial_prices


def log_returns_to_prices_with_initial(
    log_returns: np.ndarray,
    original_data_path: str = None,
    initial_prices: np.ndarray = None,
    channel_idx: int = 0,
    use_mean_initial: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert log returns to prices, using initial prices from original data if available.
    
    Args:
        log_returns: Array of shape (R, L, N) of log returns
        original_data_path: Path to original CSV with price data (optional)
        initial_prices: Pre-computed initial prices, shape (R,) (optional)
        channel_idx: Channel index to use for initial prices (0 for Open)
        use_mean_initial: If True and initial_prices not provided, use mean of available initial prices
    
    Returns:
        Tuple of (prices, initial_prices_used)
    """
    log_returns_np = np.asarray(log_returns)
    
    if log_returns_np.ndim != 3:
        raise ValueError(f"Expected 3D array (R, L, N), got {log_returns_np.ndim}D")
    
    R, L, N = log_returns_np.shape
    
    # Get initial prices
    if initial_prices is not None:
        initial_prices_used = np.asarray(initial_prices)
        if initial_prices_used.shape != (R,):
            raise ValueError(f"initial_prices shape {initial_prices_used.shape} doesn't match expected (R,)")
    elif original_data_path is not None:
        initial_prices_used = get_initial_prices_from_original_data(
            original_data_path, log_returns_np, L, channel_idx
        )
    else:
        # Use mean initial price from a representative sample, or default to 1.0
        if use_mean_initial:
            # For synthetic data, we might want to use a representative initial price
            # Use the mean of the first prices from real data, or default to 1.0
            initial_prices_used = np.ones(R) * 1.0
        else:
            initial_prices_used = np.ones(R) * 1.0
    
    # Convert to prices using the initial prices
    # For each sample r, use initial_prices_used[r] as the starting price
    prices = np.zeros((R, L + 1, N))
    
    # Set initial prices for the open channel (channel 0)
    prices[:, 0, channel_idx] = initial_prices_used
    
    # For other channels, we can use a relative scaling or same initial price
    # For simplicity, use same initial price for all channels
    for ch in range(N):
        if ch != channel_idx:
            prices[:, 0, ch] = initial_prices_used  # Or could use channel-specific initial prices
    
    # Convert log returns to prices
    for t in range(L):
        prices[:, t + 1] = prices[:, t] * np.exp(log_returns_np[:, t])
    
    # Remove the initial price row to match input shape
    prices = prices[:, 1:]
    
    return prices, initial_prices_used


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


def compute_quadratic_variation(prices: np.ndarray) -> np.ndarray:
    """
    Compute quadratic variation of price series.
    
    QVar = sum((S_{t+1} - S_t)^2) over time
    
    Args:
        prices: Price sequences of shape (R, L) or (R, L, N)
    
    Returns:
        Quadratic variation of shape (R,) or (R, N)
    """
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
    prices_real: np.ndarray, 
    prices_synthetic: np.ndarray
) -> Dict[str, float]:
    """
    Compute temporal dependency metrics: MSE between quadratic variations.
    
    Args:
        prices_real: Real price sequences, shape (R, L) or (R, L, N)
        prices_synthetic: Synthetic price sequences, shape (R, L) or (R, L, N)
    
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


def compute_covariance_matrix(prices: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix of price series.
    
    Args:
        prices: Price sequences of shape (R, L) or (R, L, N)
    
    Returns:
        Covariance matrix of shape (L, L) or (N, N) depending on input
    """
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
    prices_real: np.ndarray,
    prices_synthetic: np.ndarray
) -> Dict[str, float]:
    """
    Compute correlation structure metrics: time-averaged MSE between covariance matrices.
    
    Args:
        prices_real: Real price sequences, shape (R, L) or (R, L, N)
        prices_synthetic: Synthetic price sequences, shape (R, L) or (R, L, N)
    
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
        real_train_log_returns: np.ndarray,
        real_val_log_returns: np.ndarray,
        synthetic_train_log_returns: np.ndarray,
        seq_length: int,
        original_data_path: str = None,
        strike: float = None,
        hidden_size: int = 64,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Args:
            real_train_log_returns: Real training log returns, shape (R_real, L, N)
            real_val_log_returns: Real validation log returns, shape (R_val, L, N)
            synthetic_train_log_returns: Synthetic training log returns, shape (R_syn, L, N)
            seq_length: Sequence length L
            original_data_path: Path to original CSV file with price data (to get initial prices)
            strike: Strike price for call option (if None, will use mean of initial prices)
            hidden_size: Hidden size for hedgers
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
        """
        self.real_train_log_returns = real_train_log_returns
        self.real_val_log_returns = real_val_log_returns
        self.synthetic_train_log_returns = synthetic_train_log_returns
        self.seq_length = seq_length
        self.original_data_path = original_data_path
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Convert log returns to prices
        print("Converting log returns to prices for real training data...")
        self.real_train_prices, real_train_initial = log_returns_to_prices_with_initial(
            real_train_log_returns,
            original_data_path=original_data_path,
            channel_idx=0
        )
        self.real_train_prices = self.real_train_prices[:, :, 0]  # Extract open channel only
        
        print("Converting log returns to prices for real validation data...")
        self.real_val_prices, real_val_initial = log_returns_to_prices_with_initial(
            real_val_log_returns,
            original_data_path=original_data_path,
            channel_idx=0
        )
        self.real_val_prices = self.real_val_prices[:, :, 0]  # Extract open channel only
        
        print("Converting log returns to prices for synthetic training data...")
        # For synthetic data, use mean of real initial prices
        mean_initial = np.mean(real_train_initial)
        self.synthetic_train_prices, _ = log_returns_to_prices_with_initial(
            synthetic_train_log_returns,
            initial_prices=np.ones(synthetic_train_log_returns.shape[0]) * mean_initial,
            channel_idx=0
        )
        self.synthetic_train_prices = self.synthetic_train_prices[:, :, 0]  # Extract open channel only
        
        # Set strike price
        if strike is None:
            self.strike = mean_initial  # At-the-money option
        else:
            self.strike = strike
    
    def evaluate(self, hedger_class) -> Dict[str, Any]:
        """
        Evaluate augmented testing for a given hedger class.
        
        Args:
            hedger_class: Class of the hedger (e.g., FeedforwardDeepHedger)
        
        Returns:
            Dictionary with evaluation results
        """
        # Mix synthetic and real data (50/50)
        R_real = self.real_train_prices.shape[0]
        R_syn = self.synthetic_train_prices.shape[0]
        R_mixed = min(R_real, R_syn)
        
        # Sample equal amounts from real and synthetic
        np.random.seed(42)
        real_indices = np.random.choice(R_real, R_mixed, replace=False)
        syn_indices = np.random.choice(R_syn, R_mixed, replace=False)
        
        mixed_train_prices = np.vstack([
            self.real_train_prices[real_indices],
            self.synthetic_train_prices[syn_indices]
        ])
        np.random.shuffle(mixed_train_prices)
        
        # Train hedger on mixed data
        hedger_mixed = hedger_class(
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            strike=self.strike
        )
        # Ensure data is torch tensor
        mixed_train_prices_tensor = torch.from_numpy(mixed_train_prices).float() if isinstance(mixed_train_prices, np.ndarray) else mixed_train_prices.float()
        hedger_mixed.fit(
            mixed_train_prices_tensor,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=False
        )
        
        # Train hedger on real data only
        hedger_real = hedger_class(
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            strike=self.strike
        )
        # Ensure data is torch tensor
        real_train_prices_tensor = torch.from_numpy(self.real_train_prices).float() if isinstance(self.real_train_prices, np.ndarray) else self.real_train_prices.float()
        hedger_real.fit(
            real_train_prices_tensor,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=False
        )
        
        # Evaluate on real validation set
        val_prices_tensor = torch.from_numpy(self.real_val_prices).float() if isinstance(self.real_val_prices, np.ndarray) else self.real_val_prices.float()
        
        X_mixed = compute_X_values(hedger_mixed, val_prices_tensor, self.strike)
        X_real_only = compute_X_values(hedger_real, val_prices_tensor, self.strike)
        
        # Compute metrics
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
        
        return results


class AlgorithmComparisonEvaluator:
    """
    Algorithm Comparison: Generate train/val/test synthetic data, train 4 hedgers on both real and synthetic, evaluate.
    """
    
    def __init__(
        self,
        real_train_log_returns: np.ndarray,
        real_val_log_returns: np.ndarray,
        real_test_log_returns: np.ndarray,
        synthetic_train_log_returns: np.ndarray,
        synthetic_val_log_returns: np.ndarray,
        synthetic_test_log_returns: np.ndarray,
        seq_length: int,
        original_data_path: str = None,
        strike: float = None,
        hidden_size: int = 64,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Args:
            real_train_log_returns: Real training log returns, shape (R, L, N)
            real_val_log_returns: Real validation log returns, shape (R, L, N)
            real_test_log_returns: Real test log returns, shape (R, L, N)
            synthetic_train_log_returns: Synthetic training log returns, shape (R, L, N)
            synthetic_val_log_returns: Synthetic validation log returns, shape (R, L, N)
            synthetic_test_log_returns: Synthetic test log returns, shape (R, L, N)
            seq_length: Sequence length L
            original_data_path: Path to original CSV file with price data (to get initial prices)
            strike: Strike price for call option (if None, will use mean of initial prices)
            hidden_size: Hidden size for hedgers
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
        """
        self.seq_length = seq_length
        self.original_data_path = original_data_path
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Convert log returns to prices
        print("Converting log returns to prices for real data...")
        self.real_train_prices, real_train_initial = log_returns_to_prices_with_initial(
            real_train_log_returns,
            original_data_path=original_data_path,
            channel_idx=0
        )
        self.real_train_prices = self.real_train_prices[:, :, 0]  # Extract open channel only
        
        self.real_val_prices, _ = log_returns_to_prices_with_initial(
            real_val_log_returns,
            original_data_path=original_data_path,
            channel_idx=0
        )
        self.real_val_prices = self.real_val_prices[:, :, 0]
        
        self.real_test_prices, _ = log_returns_to_prices_with_initial(
            real_test_log_returns,
            original_data_path=original_data_path,
            channel_idx=0
        )
        self.real_test_prices = self.real_test_prices[:, :, 0]
        
        print("Converting log returns to prices for synthetic data...")
        # For synthetic data, use mean of real initial prices
        mean_initial = np.mean(real_train_initial)
        
        self.synthetic_train_prices, _ = log_returns_to_prices_with_initial(
            synthetic_train_log_returns,
            initial_prices=np.ones(synthetic_train_log_returns.shape[0]) * mean_initial,
            channel_idx=0
        )
        self.synthetic_train_prices = self.synthetic_train_prices[:, :, 0]
        
        self.synthetic_val_prices, _ = log_returns_to_prices_with_initial(
            synthetic_val_log_returns,
            initial_prices=np.ones(synthetic_val_log_returns.shape[0]) * mean_initial,
            channel_idx=0
        )
        self.synthetic_val_prices = self.synthetic_val_prices[:, :, 0]
        
        self.synthetic_test_prices, _ = log_returns_to_prices_with_initial(
            synthetic_test_log_returns,
            initial_prices=np.ones(synthetic_test_log_returns.shape[0]) * mean_initial,
            channel_idx=0
        )
        self.synthetic_test_prices = self.synthetic_test_prices[:, :, 0]
        
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
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate all four hedgers on both real and synthetic data.
        
        Returns:
            Dictionary with evaluation results for each hedger
        """
        results = {}
        
        for hedger_name, hedger_class in self.hedger_classes.items():
            print(f"Evaluating {hedger_name}...")
            
            # Train on real data
            hedger_real = hedger_class(
                seq_length=self.seq_length,
                hidden_size=self.hidden_size,
                strike=self.strike
            )
            # Ensure data is torch tensor
            real_train_prices_tensor = torch.from_numpy(self.real_train_prices).float() if isinstance(self.real_train_prices, np.ndarray) else self.real_train_prices.float()
            hedger_real.fit(
                real_train_prices_tensor,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                verbose=False
            )
            
            # Train on synthetic data
            hedger_syn = hedger_class(
                seq_length=self.seq_length,
                hidden_size=self.hidden_size,
                strike=self.strike
            )
            # Ensure data is torch tensor
            synthetic_train_prices_tensor = torch.from_numpy(self.synthetic_train_prices).float() if isinstance(self.synthetic_train_prices, np.ndarray) else self.synthetic_train_prices.float()
            hedger_syn.fit(
                synthetic_train_prices_tensor,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                verbose=False
            )
            
            # Evaluate on real test set
            real_test_prices_tensor = torch.from_numpy(self.real_test_prices).float() if isinstance(self.real_test_prices, np.ndarray) else self.real_test_prices.float()
            X_real_test = compute_X_values(
                hedger_real,
                real_test_prices_tensor,
                self.strike
            )
            
            # Evaluate on synthetic test set
            synthetic_test_prices_tensor = torch.from_numpy(self.synthetic_test_prices).float() if isinstance(self.synthetic_test_prices, np.ndarray) else self.synthetic_test_prices.float()
            X_syn_test = compute_X_values(
                hedger_syn,
                synthetic_test_prices_tensor,
                self.strike
            )
            
            # Compute metrics for real test set
            marginal_real = compute_marginal_metrics(X_real_test, X_real_test)  # Baseline
            temporal_real = compute_temporal_metrics(self.real_test_prices, self.real_test_prices)
            correlation_real = compute_correlation_metrics(self.real_test_prices, self.real_test_prices)
            
            # Compute metrics for synthetic test set
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
        
        return results


def save_utility_scores(results: Dict[str, Any], model_name: str, results_dir: Path = None):
    """
    Save utility scores to JSON file.
    
    Args:
        results: Dictionary containing evaluation results
        model_name: Name of the generative model
        results_dir: Directory to save results (default: results/utility_scores/)
    """
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
    
    print(f"Utility scores saved to {output_path}")
    return output_path

