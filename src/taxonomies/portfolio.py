"""Portfolio optimization downstream evaluation.

Implements the framework of DeMiguel, Garlappi & Uppal (2009):
"Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?"
Review of Financial Studies, 22(5), 1915–1953.

Strategies (price/return data only — no volume):
  - 1/N (Equal Weight)    — naive diversification benchmark
  - Minimum Variance (GMV) — global minimum variance portfolio
  - Mean-Variance (Tangency) — maximum Sharpe ratio portfolio
  - Mean-Variance (Long-Only) — short-sale constrained

Metrics (from the paper):
  - Out-of-sample annualized Sharpe ratio
  - Certainty-Equivalent Return (CEQ) at multiple γ
  - Portfolio Turnover
  - Herfindahl Concentration Index

Ablation: vary number of assets (5, 10, 25) to study diversification effects.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _ensure_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Compute sample covariance matrix from (T, C) return matrix."""
    T = returns.shape[0]
    mean = returns.mean(axis=0, keepdims=True)
    centered = returns - mean
    return (centered.T @ centered) / (T - 1)


def _portfolio_metrics(
    portfolio_returns: np.ndarray,
    weights_history: Optional[np.ndarray] = None,
    gamma_values: Tuple[float, ...] = (1.0, 2.0, 5.0),
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Compute standard portfolio evaluation metrics.

    Args:
        portfolio_returns: (T,) array of out-of-sample portfolio returns (per period).
        weights_history: (T, C) or (n_rebalance, C) array of weights. If None,
            turnover is not computed.
        gamma_values: Risk-aversion parameters for CEQ.
        periods_per_year: Annualization factor.

    Returns:
        Dict of metric_name → value.
    """
    r = np.asarray(portfolio_returns, dtype=np.float64)
    T = len(r)
    if T < 2:
        return {}

    mean_ret = np.mean(r)
    std_ret = np.std(r, ddof=1)

    # Annualized Sharpe ratio
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0.0

    # Certainty-Equivalent Return (annualized)
    ceq = {}
    for gamma in gamma_values:
        ceq[f"ceq_gamma_{gamma}"] = float(
            (mean_ret - (gamma / 2.0) * (std_ret ** 2)) * periods_per_year
        )

    # Maximum Drawdown
    cumulative = np.cumprod(1.0 + r)
    hwm = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - hwm) / hwm
    max_dd = float(np.min(drawdowns))

    # Calmar ratio
    annualized_return = mean_ret * periods_per_year
    calmar = annualized_return / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

    # Downside deviation (Sortino) — matches quantstats convention:
    # sqrt(mean(min(0, r)^2))
    neg = np.minimum(r, 0.0)
    downside_std = np.sqrt(np.mean(neg ** 2))
    sortino = (mean_ret / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    # Omega ratio (threshold = 0)
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    omega = float(gains / losses) if losses > 0 else float("inf")

    # Tail ratio: P95 / |P5|
    p95 = np.percentile(r, 95)
    p05 = np.percentile(r, 5)
    tail_ratio = float(p95 / abs(p05)) if abs(p05) > 1e-12 else float("inf")

    # Turnover
    turnover = 0.0
    if weights_history is not None and weights_history.shape[0] > 1:
        w = np.asarray(weights_history, dtype=np.float64)
        turnover = float(np.mean(np.sum(np.abs(w[1:] - w[:-1]), axis=1)))

    # Herfindahl concentration (last weights)
    herfindahl = 0.0
    if weights_history is not None and weights_history.shape[0] > 0:
        w_last = np.asarray(weights_history[-1], dtype=np.float64)
        herfindahl = float(np.sum(w_last ** 2))

    return {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "omega_ratio": omega,
        "tail_ratio": tail_ratio,
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(std_ret * np.sqrt(periods_per_year)),
        "turnover": turnover,
        "herfindahl": herfindahl,
        **ceq,
    }


# ---------------------------------------------------------------------------
# Portfolio strategies
# ---------------------------------------------------------------------------


def _minimum_variance_weights(cov: np.ndarray, allow_short: bool = True) -> np.ndarray:
    """Global Minimum Variance (GMV) portfolio weights.

    Solves: min_w w'Σw  s.t. w'1 = 1  (and w ≥ 0 if long-only).
    Closed-form for unconstrained: w* = Σ^{-1}1 / (1'Σ^{-1}1).
    """
    C = cov.shape[0]
    cov = np.asarray(cov, dtype=np.float64)

    if allow_short:
        try:
            inv_cov = np.linalg.pinv(cov)  # pseudoinverse for numerical stability
            ones = np.ones((C, 1))
            w = inv_cov @ ones
            w = w / (ones.T @ w)
            return w.flatten()
        except Exception:
            return np.ones(C) / C

    # Long-only GMV via quadratic programming
    try:
        from scipy.optimize import minimize

        def objective(w):
            return w @ cov @ w

        w0 = np.ones(C) / C
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(C)]
        result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            return result.x
    except Exception:
        pass
    return np.ones(C) / C


def _tangency_weights(cov: np.ndarray, mean_ret: np.ndarray, allow_short: bool = True, rf: float = 0.0) -> np.ndarray:
    """Maximum Sharpe ratio (tangency) portfolio weights.

    Solves: max_w (w'μ - rf) / sqrt(w'Σw)  s.t. w'1 = 1.
    Unconstrained closed-form: w* ∝ Σ^{-1}(μ - rf·1).
    """
    C = cov.shape[0]
    cov = np.asarray(cov, dtype=np.float64)
    mean_ret = np.asarray(mean_ret, dtype=np.float64).flatten()

    excess = mean_ret - rf

    if allow_short:
        try:
            inv_cov = np.linalg.pinv(cov)  # pseudoinverse for numerical stability
            w = inv_cov @ excess
            w = w / np.sum(w)
            return w
        except Exception:
            return np.ones(C) / C

    # Long-only tangency via QP
    try:
        from scipy.optimize import minimize

        def neg_sharpe(w):
            port_ret = w @ mean_ret
            port_vol = np.sqrt(max(w @ cov @ w, 1e-15))
            return -(port_ret - rf) / port_vol

        w0 = np.ones(C) / C
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(C)]
        result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            return result.x
    except Exception:
        pass
    return np.ones(C) / C


def _equal_weights(C: int) -> np.ndarray:
    """1/N portfolio weights."""
    return np.ones(C) / C


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class PortfolioOptimizationEvaluator:
    """Evaluate synthetic data utility via portfolio optimization (DeMiguel et al. 2009).

    Estimates covariance (and optionally mean) from synthetic returns, then evaluates
    the resulting portfolio weights on real out-of-sample test returns using a
    rolling-window or single-split approach.
    """

    def __init__(
        self,
        real_test_returns: np.ndarray,
        synthetic_returns: np.ndarray,
        estimation_window: int = 252,
        rebalance_freq: int = 21,
        periods_per_year: int = 252,
        rf: float = 0.0,
        num_assets_ablation: Optional[List[int]] = None,
    ):
        """
        Args:
            real_test_returns: (T, C) out-of-sample real returns (log returns).
            synthetic_returns: (R, L, C) synthetic log-return samples.
            estimation_window: Rolling window length for covariance estimation.
            rebalance_freq: Rebalance every N periods.
            periods_per_year: For annualization.
            rf: Risk-free rate.
            num_assets_ablation: List of asset counts to evaluate (e.g., [5, 10, 25]).
        """
        self.real_test = _ensure_numpy(real_test_returns)
        if self.real_test.ndim == 3:
            # (N, L, C) → flatten samples into (N*L, C) for estimation
            self.real_test = self.real_test.reshape(-1, self.real_test.shape[-1])
        if self.real_test.ndim != 2:
            raise ValueError(f"Expected (T,C) real_test_returns, got {self.real_test.shape}")

        self.synthetic = _ensure_numpy(synthetic_returns)
        if self.synthetic.ndim == 3:
            self.synthetic = self.synthetic.reshape(-1, self.synthetic.shape[-1])

        # Adapt estimation window to synthetic sequence length
        synth_len = self.synthetic.shape[0] if self.synthetic.ndim == 2 else self.synthetic.shape[1]
        self.estimation_window = min(estimation_window, max(synth_len, 21))
        self.rebalance_freq = min(rebalance_freq, max(synth_len // 4, 1))
        self.periods_per_year = periods_per_year
        self.rf = rf
        self.num_assets_ablation = num_assets_ablation or [5, 10, self.real_test.shape[1]]

    def evaluate(self) -> Dict:
        """Run portfolio optimization evaluation for all strategies and ablations."""
        total_assets = self.real_test.shape[1]
        asset_counts = sorted(set(min(c, total_assets) for c in self.num_assets_ablation))

        all_results = {}
        for C in asset_counts:
            # Select first C assets
            real = self.real_test[:, :C]
            synth = self.synthetic[:, :C]
            all_results[f"n_assets_{C}"] = self._evaluate_fixed_assets(real, synth, C)
        return all_results

    def _evaluate_fixed_assets(
        self, real: np.ndarray, synth: np.ndarray, C: int
    ) -> Dict:
        """Evaluate all strategies for a fixed set of C assets."""
        # Estimate covariance and mean from synthetic data
        synth_cov = _sample_covariance(synth)
        synth_mean = synth.mean(axis=0)

        strategies = {
            "1N": _equal_weights(C),
            "GMV": _minimum_variance_weights(synth_cov, allow_short=True),
            "GMV_LongOnly": _minimum_variance_weights(synth_cov, allow_short=False),
            "Tangency": _tangency_weights(synth_cov, synth_mean, allow_short=True, rf=self.rf),
            "Tangency_LongOnly": _tangency_weights(synth_cov, synth_mean, allow_short=False, rf=self.rf),
        }

        results = {}
        for name, weights in strategies.items():
            weights = weights / weights.sum()  # Ensure sum-to-1
            w = np.asarray(weights, dtype=np.float64)

            # Out-of-sample evaluation: apply fixed weights to real returns
            port_returns = real @ w  # (T,)
            metrics = _portfolio_metrics(
                port_returns, weights_history=w[np.newaxis, :],
                periods_per_year=self.periods_per_year,
            )
            results[name] = {**metrics, "weights": w.tolist()}

        # Add ground-truth comparison: estimate from real data
        real_cov = _sample_covariance(real)
        real_mean = real.mean(axis=0)
        for name_suffix, use_short, fn in [
            ("_Real_GMV", True, _minimum_variance_weights),
            ("_Real_GMV_LongOnly", False, _minimum_variance_weights),
            ("_Real_Tangency", True, _tangency_weights),
            ("_Real_Tangency_LongOnly", False, _tangency_weights),
        ]:
            if "GMV" in name_suffix:
                w_real = fn(real_cov, allow_short=use_short)
            else:
                w_real = fn(real_cov, real_mean, allow_short=use_short, rf=self.rf)
            w_real = w_real / w_real.sum()
            port_returns = real @ w_real
            results[f"Real_GT_{name_suffix.split('_')[1]}{name_suffix.split('_')[2]}"] = {
                **_portfolio_metrics(port_returns, weights_history=w_real[np.newaxis, :],
                                    periods_per_year=self.periods_per_year),
                "weights": w_real.tolist(),
            }

        return results
