"""P&L (Profit & Loss) evaluation for synthetic financial time series.

Evaluates the financial utility of synthetic data by simulating a simple
trading strategy on both real and synthetic return paths, then comparing
the resulting P&L distributions using standard quantitative finance metrics.

Methodology:
  1. Convert log-returns to price paths via cumulative product.
  2. Apply a simple strategy (e.g., buy-and-hold on each asset, or equal-weight).
  3. Compute P&L statistics: Sharpe, Sortino, Max Drawdown, Calmar, Omega, Tail ratio.
  4. Compare the P&L distribution from real vs synthetic paths.

References:
  - Standard QF performance evaluation: Sharpe (1966), Sortino & Price (1994).
  - Synthetic data backtesting: Lopez de Prado (2018), "Advances in Financial ML".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _ensure_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _accumulate_returns(log_returns: np.ndarray, initial_value: float = 1.0) -> np.ndarray:
    """Convert log-return series to cumulative price/NAV.

    Args:
        log_returns: (R, L) or (R, L, C) log returns.
        initial_value: Starting NAV (default 1.0).

    Returns:
        Cumulative price/NAV paths (same shape as input).
    """
    r = np.asarray(log_returns, dtype=np.float64)
    return initial_value * np.exp(np.cumsum(r, axis=1))


def _log_returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from price paths."""
    return np.diff(np.log(prices), axis=1)


def _pnl_metrics(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Compute standard P&L evaluation metrics for a return series.

    Args:
        returns: (T,) or (R, T) array of period log returns.
            If 2D, metrics are averaged across paths.
        periods_per_year: Annualization factor.

    Returns:
        Dict of aggregated metrics.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        r = r.reshape(1, -1)

    R, T = r.shape
    if T < 2:
        return {}

    mean_ret = np.mean(r, axis=1)
    std_ret = np.std(r, axis=1, ddof=1)

    # Annualized Sharpe
    sharpe = np.mean(mean_ret / (std_ret + 1e-15)) * np.sqrt(periods_per_year)

    # Annualized return
    ann_return = np.mean(mean_ret) * periods_per_year

    # Annualized vol
    ann_vol = np.mean(std_ret) * np.sqrt(periods_per_year)

    # Maximum Drawdown (per path, then average)
    max_dds = []
    for i in range(R):
        cumulative = np.cumprod(1.0 + r[i])
        hwm = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - hwm) / (hwm + 1e-15)
        max_dds.append(float(np.min(drawdowns)))
    max_dd = float(np.mean(max_dds))

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

    # Sortino ratio (matching quantstats: sqrt(mean(min(0,r)^2)))
    sortinos = []
    for i in range(R):
        neg = np.minimum(r[i], 0.0)
        d_std = np.sqrt(np.mean(neg ** 2))
        sortinos.append(mean_ret[i] / (d_std + 1e-15))
    sortino = np.mean(sortinos) * np.sqrt(periods_per_year)

    # Omega ratio (threshold = 0)
    omegas = []
    for i in range(R):
        gains = r[i][r[i] > 0].sum()
        losses = abs(r[i][r[i] < 0].sum())
        omegas.append(float(gains / losses) if losses > 1e-15 else float("inf"))
    omega = float(np.mean([o for o in omegas if np.isfinite(o)])) if omegas else 0.0

    # Tail ratio: P95 / |P5| (averaged)
    tail_ratios = []
    for i in range(R):
        p95 = np.percentile(r[i], 95)
        p05 = np.percentile(r[i], 5)
        tr = p95 / abs(p05) if abs(p05) > 1e-12 else float("inf")
        tail_ratios.append(tr)
    tail_ratio = float(np.mean([t for t in tail_ratios if np.isfinite(t)]))

    # Terminal wealth
    terminal_wealth = np.prod(1.0 + r, axis=1)

    return {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "omega_ratio": omega,
        "tail_ratio": tail_ratio,
        "annualized_return": float(ann_return),
        "annualized_volatility": float(ann_vol),
        "terminal_wealth_mean": float(np.mean(terminal_wealth)),
        "terminal_wealth_std": float(np.std(terminal_wealth)),
        "terminal_wealth_min": float(np.min(terminal_wealth)),
        "terminal_wealth_max": float(np.max(terminal_wealth)),
        "prob_positive_return": float(np.mean(terminal_wealth > 1.0)),
    }


class PnLEvaluator:
    """Profit & Loss evaluation for comparing real vs synthetic return paths.

    Simulates an equal-weight buy-and-hold strategy on each asset independently,
    then compares the P&L distributions from real and synthetic return data.
    """

    def __init__(
        self,
        real_log_returns: np.ndarray,
        synthetic_log_returns: np.ndarray,
        strategy: str = "equal_weight_bh",
        periods_per_year: int = 252,
        initial_value: float = 1.0,
    ):
        """
        Args:
            real_log_returns: (T, C) or (N, L, C) real log returns.
            synthetic_log_returns: (R, L, C) synthetic log returns.
            strategy: 'equal_weight_bh' (equal-weight buy-and-hold),
                      'per_asset' (per-asset buy-and-hold).
            periods_per_year: Annualization factor.
            initial_value: Starting NAV.
        """
        self.real = _ensure_numpy(real_log_returns)
        self.synthetic = _ensure_numpy(synthetic_log_returns)
        self.strategy = strategy
        self.periods_per_year = periods_per_year
        self.initial_value = float(initial_value)

        # Standardize to 3D
        if self.real.ndim == 2:
            self.real = self.real[np.newaxis, :, :]
        if self.synthetic.ndim == 2:
            self.synthetic = self.synthetic[np.newaxis, :, :]

    def evaluate(self) -> Dict:
        """Run P&L evaluation and compare real vs synthetic distributions."""
        if self.strategy == "equal_weight_bh":
            return self._evaluate_equal_weight()
        return self._evaluate_per_asset()

    def _evaluate_equal_weight(self) -> Dict:
        """Equal-weight buy-and-hold strategy."""
        C = self.real.shape[2]

        # Equal weight across all assets
        weights = np.ones(C) / C

        # Real paths: compute portfolio returns
        real_port_ret = np.sum(self.real * weights, axis=2)  # (N, L)
        real_metrics = _pnl_metrics(real_port_ret, self.periods_per_year)

        # Synthetic paths: compute portfolio returns
        syn_port_ret = np.sum(self.synthetic * weights, axis=2)  # (R, L)
        synth_metrics = _pnl_metrics(syn_port_ret, self.periods_per_year)

        # Compute pairwise distribution distances
        real_cum = _accumulate_returns(real_port_ret, self.initial_value)
        synth_cum = _accumulate_returns(syn_port_ret, self.initial_value)

        # Terminal wealth distribution comparison (Wasserstein-1 distance)
        from scipy.stats import wasserstein_distance

        real_terminal = real_cum[:, -1].flatten()
        synth_terminal = synth_cum[:, -1].flatten()
        tw_wasserstein = float(wasserstein_distance(real_terminal, synth_terminal))

        return {
            "real": real_metrics,
            "synthetic": synth_metrics,
            "terminal_wealth_wasserstein": tw_wasserstein,
            "strategy": "equal_weight_bh",
            "strategy_weights": weights.tolist(),
        }

    def _evaluate_per_asset(self) -> Dict:
        """Per-asset buy-and-hold: evaluate each asset independently."""
        C = self.real.shape[2]
        per_asset = {}
        for c in range(C):
            real_c = self.real[:, :, c]
            synth_c = self.synthetic[:, :, c]
            per_asset[f"asset_{c}"] = {
                "real": _pnl_metrics(real_c, self.periods_per_year),
                "synthetic": _pnl_metrics(synth_c, self.periods_per_year),
            }
        return {"per_asset": per_asset, "strategy": "per_asset_bh"}
