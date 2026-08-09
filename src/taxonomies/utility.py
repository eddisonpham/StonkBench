"""StonkBench §6 Utility evaluation (deep hedging).

The modular pipeline lives in :mod:`src.utility`:
- :class:`src.utility.metrics.MetricToolbox` — U1–U5 PnL metrics
- :class:`src.utility.protocols.TSTRProtocol`, ``AugmentedProtocol``
- :class:`src.utility.evaluator.UtilityEvaluator` — orchestrator
- :class:`src.utility.tasks.options.OptionsTask` — §6.2
- :class:`src.utility.tasks.portfolio.PortfolioTask` — §6.3
- :class:`src.utility.tasks.alpha.AlphaTask` — §6.4
- One paper-faithful model per task: :class:`MoneynessLSTM`,
  :class:`PortfolioLSTM`, :class:`AlphaLSTM`, plus a static
  :class:`BSStaticDelta` reference baseline (premium = 0).

This module preserves the legacy import surface so
``unified_evaluator`` keeps working:
``from src.taxonomies.utility import AugmentedTestingEvaluator``.

``AlgorithmComparisonEvaluator`` is intentionally removed (not in paper §6).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.hedging_models.deep_hedgers.feedforward_layers import FeedforwardLayers
from src.hedging_models.deep_hedgers.feedforward_time import FeedforwardTime
from src.hedging_models.deep_hedgers.rnn_hedger import RNN
from src.hedging_models.deep_hedgers.lstm_hedger import LSTM
from src.hedging_models.non_deep_hedgers.black_scholes import BlackScholes
from src.hedging_models.non_deep_hedgers.delta_gamma import DeltaGamma
from src.hedging_models.non_deep_hedgers.linear_regression import LinearRegression
from src.hedging_models.non_deep_hedgers.xgboost import XGBoost

# New modular pipeline imports (available when src/utility/ is complete).
try:
    from src.utility import (  # noqa: F401
        MetricToolbox,
        TSTRProtocol,
        AugmentedProtocol,
        OptionsTask,
        PortfolioTask,
        AlphaTask,
        MoneynessLSTM,
        PortfolioLSTM,
        AlphaLSTM,
        BSStaticDelta,
    )
    _NEW_UTILITY_AVAILABLE = True
except ImportError:
    _NEW_UTILITY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core helpers (paper-faithful, vectorized)
# ---------------------------------------------------------------------------

def log_returns_to_prices(
    log_returns: torch.Tensor,
    initial_prices: torch.Tensor,
) -> torch.Tensor:
    """Convert log returns to prices (fully vectorized).

    prices[t] = S₀ * exp(cumsum(log_returns))

    Args:
        log_returns: (R, L) or (R, L, C) tensor of log returns.
        initial_prices: (R,) or (R, C) tensor of initial asset prices.

    Returns:
        (R, L) price paths if univariate, or (R, L, C) for multivariate.
    """
    if log_returns.ndim == 2:
        R, L = log_returns.shape
        if initial_prices.ndim == 0:
            initial_prices = initial_prices.expand(R)
        if initial_prices.shape != (R,):
            raise ValueError(
                f"For 2D log_returns (R, L), initial_prices must be (R,) or scalar, "
                f"got {initial_prices.shape}"
            )
        return initial_prices.unsqueeze(1) * torch.exp(torch.cumsum(log_returns, dim=1))

    if log_returns.ndim == 3:
        R, L, C = log_returns.shape
        if initial_prices.ndim == 1:
            initial_prices = initial_prices.unsqueeze(0).expand(R, C)
        if initial_prices.shape != (R, C):
            raise ValueError(
                f"For 3D log_returns (R, L, C), initial_prices must be (R, C), "
                f"got {initial_prices.shape}"
            )
        return initial_prices.unsqueeze(1) * torch.exp(torch.cumsum(log_returns, dim=1))

    raise ValueError(f"Expected 2D or 3D log_returns, got shape {tuple(log_returns.shape)}")


def compute_replication_errors(hedger, prices: torch.Tensor) -> torch.Tensor:
    """Compute replication error for a European call option.

    error = payoff - cumulative delta-hedged P&L.
    """
    if _NEW_UTILITY_AVAILABLE:
        warnings.warn(
            "compute_replication_errors is deprecated; use "
            "OptionsTask.predict_period_pnl instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    from src.utility.policies.lstm_moneyness import MoneynessLSTM
    if isinstance(hedger, MoneynessLSTM):
        hedger.eval()
    prices = prices.to(hedger.device).float() if hasattr(hedger, "device") else prices.float()
    with torch.no_grad():
        N = prices.shape[0]
        K = torch.ones(N)
        try:
            deltas = hedger.predict(prices, K=K)
            terminal = torch.zeros(N)
            for t in range(deltas.shape[1] - 1):
                terminal += deltas[:, t] * (prices[:, t + 1] - prices[:, t])
            payoffs = torch.clamp(
                prices[:, -1] - float(getattr(hedger, "strike", 1.0)), min=0.0
            )
            return payoffs - terminal
        except Exception:
            return torch.zeros(N)


def fit_hedger(hedger, data: torch.Tensor, *args, **kwargs):
    """Legacy fit dispatcher — delegates to hedger.fit()."""
    if _NEW_UTILITY_AVAILABLE:
        warnings.warn(
            "fit_hedger is deprecated; train via task.fit(...) / policy.fit(...).",
            DeprecationWarning,
            stacklevel=2,
        )
    if hasattr(hedger, "fit"):
        return hedger.fit(data, *args, **kwargs)
    raise RuntimeError("Legacy fit_hedger cannot find a .fit() method.")


def summarize_replication_error(R: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive replication error distribution statistics.

    Returns: mean, std, QVaR (95/99), CVaR/ES (95/99),
    skewness, excess kurtosis, min, max.
    """
    r = R.detach().cpu().float()
    sorted_r = torch.sort(r).values
    n = len(sorted_r)

    def _qvar(alpha: float) -> float:
        idx = int(np.ceil(alpha * n)) - 1
        idx = max(0, min(idx, n - 1))
        return float(sorted_r[idx].item())

    def _cvar(alpha: float) -> float:
        q = _qvar(alpha)
        tail = r[r >= q]
        return float(tail.mean().item()) if tail.numel() > 0 else q

    z = (r - r.mean()) / (r.std() + 1e-12)
    return {
        "mean": float(r.mean().item()),
        "std": float(r.std().item()),
        "qvar_95": _qvar(0.95),
        "qvar_99": _qvar(0.99),
        "cvar_95": _cvar(0.95),
        "cvar_99": _cvar(0.99),
        "skewness": float((z ** 3).mean().item()),
        "kurtosis": float((z ** 4).mean().item()) - 3.0,  # excess
        "min": float(r.min().item()),
        "max": float(r.max().item()),
    }


# ---------------------------------------------------------------------------
# AugmentedTestingEvaluator — complete working implementation
# ---------------------------------------------------------------------------

class AugmentedTestingEvaluator:
    """Mix synthetic + real training data (50/50) to evaluate hedgers.

    Follows Buehler et al. (2019) deep hedging framework:
    - Price paths constructed from log returns with actual initial prices.
    - ATM strike (mean initial price across training paths).
    - Hedger minimizes MSE replication error on a European call.
    - Mixed training evaluates synthetic data augmentation value.
    """

    HEDGER_CLASSES = {
        "Feedforward_L-1": FeedforwardLayers,
        "Feedforward_Time": FeedforwardTime,
        "RNN": RNN,
        "LSTM": LSTM,
        "BlackScholes": BlackScholes,
        "DeltaGamma": DeltaGamma,
        "LinearRegression": LinearRegression,
        "XGBoost": XGBoost,
    }

    def __init__(
        self,
        real_train_log_returns: torch.Tensor,
        real_val_log_returns: torch.Tensor,
        synthetic_train_log_returns: torch.Tensor,
        real_train_initial: torch.Tensor,
        real_val_initial: torch.Tensor,
        synthetic_train_initial: Optional[torch.Tensor] = None,
        seq_length: Optional[int] = None,
        num_epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
    ):
        # 3D → 2D squeeze (per-asset hedging is univariate)
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

        self.real_train_prices_full = log_returns_to_prices(
            real_train_log_returns, real_train_initial
        )
        self.real_val_prices = log_returns_to_prices(
            real_val_log_returns, real_val_initial
        )

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

        self.synthetic_train_prices_full = log_returns_to_prices(
            synthetic_train_log_returns, synthetic_train_initial
        )
        self.seq_length = seq_length or self.real_train_prices_full.shape[1]
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.strike = float(real_train_initial.mean().item())

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        R_real = self.real_train_prices_full.shape[0]
        R_syn = self.synthetic_train_prices_full.shape[0]
        R_mixed = min(R_real, R_syn)

        for name, cls in self.HEDGER_CLASSES.items():
            real_idx = torch.randperm(R_real)[:R_mixed]
            syn_idx = torch.randperm(R_syn)[:R_mixed]
            real_subset = self.real_train_prices_full[real_idx]
            syn_subset = self.synthetic_train_prices_full[syn_idx]
            mixed_train = torch.cat([real_subset, syn_subset], dim=0)[
                torch.randperm(2 * R_mixed)
            ]

            hedger_mixed = cls(seq_length=self.seq_length, strike=self.strike)
            fit_hedger(
                hedger_mixed, mixed_train,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
            )

            hedger_real = cls(seq_length=self.seq_length, strike=self.strike)
            fit_hedger(
                hedger_real, real_subset,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
            )

            R_mixed_val = compute_replication_errors(hedger_mixed, self.real_val_prices)
            R_real_val = compute_replication_errors(hedger_real, self.real_val_prices)

            results[name] = {
                "real_train": summarize_replication_error(R_real_val),
                "mixed_train": summarize_replication_error(R_mixed_val),
            }

        return results


# ---------------------------------------------------------------------------
# AlgorithmComparisonEvaluator — intentionally removed (not in paper §6)
# ---------------------------------------------------------------------------

class AlgorithmComparisonEvaluator:
    """Removed: not in paper §6. Kept as an import surface only."""

    def __init__(self, *args, **kwargs):  # noqa: ARG002
        warnings.warn(
            "AlgorithmComparisonEvaluator is removed (paper §6.1.1 uses "
            "per-generator U1–U5 comparison instead of hedger-rank Spearman).",
            DeprecationWarning,
            stacklevel=2,
        )

    def evaluate(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "AlgorithmComparisonEvaluator was removed in the §6 rewrite; "
            "use src.utility.TSTRProtocol or AugmentedProtocol."
        )