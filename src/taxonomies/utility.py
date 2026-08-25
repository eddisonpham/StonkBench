import numpy as np
import torch
from typing import Dict, Any

from src.hedging_models.base_hedger import DeepHedgingModel, NonDeepHedgingModel
from src.hedging_models.deep_hedgers.feedforward_layers import FeedforwardLayers
from src.hedging_models.deep_hedgers.feedforward_time import FeedforwardTime
from src.hedging_models.deep_hedgers.rnn_hedger import RNN
from src.hedging_models.deep_hedgers.lstm_hedger import LSTM
from src.hedging_models.non_deep_hedgers.black_scholes import BlackScholes
from src.hedging_models.non_deep_hedgers.delta_gamma import DeltaGamma
from src.hedging_models.non_deep_hedgers.linear_regression import LinearRegression
from src.hedging_models.non_deep_hedgers.xgboost import XGBoost


def log_returns_to_prices(
    log_returns: torch.Tensor,
    initial_prices: torch.Tensor,
) -> torch.Tensor:
    """Convert log returns to prices using initial prices (fully vectorized).

    prices[t] = S₀ * exp(cumsum(log_returns))

    Args:
        log_returns: (R, L) or (R, L, C) tensor of log returns.
        initial_prices: (R,) or (R, C) tensor of initial asset prices (NOT log returns).

    Returns:
        (R, L) price paths if univariate input, or (R, L, C) price paths for multivariate.
    """
    if log_returns.ndim == 2:
        # Univariate: (R, L)
        R, L = log_returns.shape
        if initial_prices.ndim == 0:
            initial_prices = initial_prices.expand(R)
        if initial_prices.shape != (R,):
            raise ValueError(
                f"For 2D log_returns (R, L), initial_prices must be (R,) or scalar, got {initial_prices.shape}"
            )
        return initial_prices.unsqueeze(1) * torch.exp(torch.cumsum(log_returns, dim=1))

    if log_returns.ndim == 3:
        # Multivariate: (R, L, C)
        R, L, C = log_returns.shape
        if initial_prices.ndim == 1:
            initial_prices = initial_prices.unsqueeze(0).expand(R, C)
        if initial_prices.shape != (R, C):
            raise ValueError(
                f"For 3D log_returns (R, L, C), initial_prices must be (R, C), got {initial_prices.shape}"
            )
        return initial_prices.unsqueeze(1) * torch.exp(torch.cumsum(log_returns, dim=1))

    raise ValueError(f"Expected 2D or 3D log_returns, got shape {tuple(log_returns.shape)}")


def compute_replication_errors(hedger, prices: torch.Tensor) -> torch.Tensor:
    """
    Compute replication errors: R = Final Payoff - Terminal Value
    for each sample path.
    """
    if isinstance(hedger, DeepHedgingModel):
        hedger.eval()
    prices = prices.to(hedger.device).float()
    with torch.no_grad():
        deltas = hedger.forward(prices)
        terminal_values = hedger.compute_terminal_value(prices, deltas)
        final_prices = prices[:, -1]
        payoffs = torch.clamp(final_prices - float(hedger.strike), min=0.0)  # European call
        R = payoffs - terminal_values
    return R


def fit_hedger(
    hedger,
    data: torch.Tensor,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3
):
    """Train a hedger (DeepHedgingModel or NonDeepHedgingModel)."""
    if isinstance(hedger, DeepHedgingModel):
        hedger.fit(data, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
    elif isinstance(hedger, NonDeepHedgingModel):
        hedger.fit(data)
    else:
        raise ValueError(f"Unknown hedger type: {type(hedger)}")


def summarize_replication_error(R: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive replication error distribution statistics.

    Returns: mean, std, QVaR at 95% and 99%, CVaR (Expected Shortfall),
    skewness, kurtosis, and min/max.
    """
    r = R.detach().cpu().float()
    sorted_r = torch.sort(r).values
    n = len(sorted_r)

    def _qvar(alpha: float) -> float:
        """Quantile Value at Risk: α-quantile of the error distribution.
        For hedging, negative errors = profit, positive = loss."""
        idx = int(np.ceil(alpha * n)) - 1
        idx = max(0, min(idx, n - 1))
        return float(sorted_r[idx].item())

    def _cvar(alpha: float) -> float:
        """Conditional VaR / Expected Shortfall: mean of errors exceeding QVaR α."""
        q = _qvar(alpha)
        tail = r[r >= q]
        return float(tail.mean().item()) if tail.numel() > 0 else q

    z = (r - r.mean()) / (r.std() + 1e-12)
    return {
        'mean': float(r.mean().item()),
        'std': float(r.std().item()),
        'qvar_95': _qvar(0.95),
        'qvar_99': _qvar(0.99),
        'cvar_95': _cvar(0.95),
        'cvar_99': _cvar(0.99),
        'skewness': float((z ** 3).mean().item()),
        'kurtosis': float((z ** 4).mean().item()) - 3.0,  # excess kurtosis
        'min': float(r.min().item()),
        'max': float(r.max().item()),
    }


class AugmentedTestingEvaluator:
    """
    Mix synthetic and real training data (50/50) to evaluate hedgers
    based on replication error on the real validation set.

    Follows the deep hedging framework of Buehler et al. (2019):
    - Price paths are constructed from log returns using actual initial prices.
    - Strike is set to the ATM level (mean initial price across training paths).
    - The hedger minimizes MSE of replication error on a European call option.
    - Mixed training evaluates whether synthetic data augments real data usefully.
    """
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
        # Convert 3D → 2D if needed (AugmentedTestingEvaluator is univariate per asset).
        if real_train_log_returns.ndim == 3:
            real_train_log_returns = real_train_log_returns[:, :, 0]
        if real_val_log_returns.ndim == 3:
            real_val_log_returns = real_val_log_returns[:, :, 0]
        if synthetic_train_log_returns.ndim == 3:
            synthetic_train_log_returns = synthetic_train_log_returns[:, :, 0]

        if real_train_initial.ndim > 1:
            real_train_initial = real_train_initial[:, 0]
        if real_val_initial.ndim > 1 and real_val_initial.shape[0]:
            real_val_initial = real_val_initial[:, 0]

        self.real_train_prices_full = log_returns_to_prices(real_train_log_returns, real_train_initial)
        self.real_val_prices = log_returns_to_prices(real_val_log_returns, real_val_initial)

        if synthetic_train_initial is None:
            mean_initial = float(real_train_initial.mean().item())
            synthetic_train_initial = torch.full(
                (synthetic_train_log_returns.shape[0],),
                mean_initial,
                device=real_train_initial.device,
                dtype=real_train_initial.dtype,
            )
        elif synthetic_train_initial.ndim > 1:
            synthetic_train_initial = synthetic_train_initial[:, 0]

        self.synthetic_train_prices_full = log_returns_to_prices(synthetic_train_log_returns, synthetic_train_initial)
        self.seq_length = seq_length or self.real_train_prices_full.shape[1]
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # ATM strike: S₀ is the mean initial price across real training paths.
        self.strike = float(real_train_initial.mean().item())

        self.hedger_classes = {
            'Feedforward_L-1': FeedforwardLayers,
            'Feedforward_Time': FeedforwardTime,
            'RNN': RNN,
            'LSTM': LSTM,
            'BlackScholes': BlackScholes,
            'DeltaGamma': DeltaGamma,
            'LinearRegression': LinearRegression,
            'XGBoost': XGBoost,
        }

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        results = {}
        R_real, R_syn = self.real_train_prices_full.shape[0], self.synthetic_train_prices_full.shape[0]
        R_mixed = min(R_real, R_syn)

        for name, cls in self.hedger_classes.items():
            # Sample balanced subsets for mixed training
            real_idx = torch.randperm(R_real)[:R_mixed]
            syn_idx = torch.randperm(R_syn)[:R_mixed]
            real_subset = self.real_train_prices_full[real_idx]
            syn_subset = self.synthetic_train_prices_full[syn_idx]
            mixed_train = torch.cat([real_subset, syn_subset], dim=0)[torch.randperm(2 * R_mixed)]

            # Train hedgers
            hedger_mixed = cls(seq_length=self.seq_length, strike=self.strike)
            fit_hedger(hedger_mixed, mixed_train, num_epochs=self.num_epochs, batch_size=self.batch_size, learning_rate=self.learning_rate)

            hedger_real = cls(seq_length=self.seq_length, strike=self.strike)
            fit_hedger(hedger_real, real_subset, num_epochs=self.num_epochs, batch_size=self.batch_size, learning_rate=self.learning_rate)

            # Compute replication error on validation set
            R_mixed_val = compute_replication_errors(hedger_mixed, self.real_val_prices)
            R_real_val = compute_replication_errors(hedger_real, self.real_val_prices)

            results[name] = {
                'real_train': summarize_replication_error(R_real_val),
                'mixed_train': summarize_replication_error(R_mixed_val)
            }

        return results


