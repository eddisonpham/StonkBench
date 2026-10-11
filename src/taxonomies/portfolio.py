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

import warnings
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

    Notes on conventions (kept identical to ``src.taxonomies.pnl``):

    * ``portfolio_returns`` are **log** returns, so wealth compounds as
      ``exp(cumsum(r))``.  It used to be ``cumprod(1 + r)``, i.e. the simple-return
      convention, which understated every drawdown on a log-return input.
    * A path with zero downside deviation has an *undefined* Sortino ratio; it is
      reported as ``inf`` (positive mean) or ``nan`` instead of being silently
      folded into ``0.0``.
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

    # Maximum Drawdown on the log-return wealth path
    cumulative = np.exp(np.cumsum(r))
    hwm = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - hwm) / hwm
    max_dd = float(np.min(drawdowns))

    # Calmar ratio
    annualized_return = mean_ret * periods_per_year
    calmar = annualized_return / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

    # Downside deviation (Sortino) — quantstats convention:
    # sqrt(mean(min(0, r)^2)).  With no downside the ratio is undefined, so it is
    # reported as inf/nan rather than the misleading 0.0 it used to return.
    neg = np.minimum(r, 0.0)
    downside_std = float(np.sqrt(np.mean(neg ** 2)))
    if downside_std > 0:
        sortino = (mean_ret / downside_std) * np.sqrt(periods_per_year)
    else:
        sortino = float("inf") if mean_ret > 0 else float("nan")

    # Omega ratio (threshold = 0)
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    omega = float(gains / losses) if losses > 0 else float("inf")

    # Tail ratio: P95 / |P5|
    p95 = np.percentile(r, 95)
    p05 = np.percentile(r, 5)
    tail_ratio = float(p95 / abs(p05)) if abs(p05) > 1e-12 else float("inf")

    # Turnover
    # Turnover requires a genuine weight *history*.  With a single static weight
    # vector there is no rebalancing and the quantity is undefined, not zero --
    # reporting 0.0 there made every entry look like a free lunch.
    turnover = None
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


def _warm_starts(cov: np.ndarray, C: int) -> List[np.ndarray]:
    """Feasible (simplex) starting points for the long-only optimisers.

    SLSQP is started at 1/N by default, but for a covariance whose entries are
    ~1e-4 the objective (~1e-5) is already below the default step tolerance, so
    the solver reports success after one iteration **without moving** and the
    caller silently accepted 1/N.  Starting from several informative points and
    keeping the best feasible candidate makes that failure impossible to miss.
    """
    starts = [np.ones(C) / C]
    diag = np.clip(np.diag(cov), 1e-18, None)
    inv_var = (1.0 / diag)
    starts.append(inv_var / inv_var.sum())
    try:
        pinv = np.linalg.pinv(cov)
        w = pinv @ np.ones(C)
        if np.isfinite(w).all() and abs(w.sum()) > 1e-15:
            w = w / w.sum()
            starts.append(np.clip(w, 0.0, None) / max(np.clip(w, 0.0, None).sum(), 1e-15))
    except Exception:
        pass
    out = []
    for s in starts:
        s = np.asarray(s, dtype=np.float64)
        if np.isfinite(s).all() and s.sum() > 0:
            out.append(np.clip(s, 0.0, 1.0) / np.clip(s, 0.0, 1.0).sum())
    return out or [np.ones(C) / C]


def _minimise_simplex(objective, C: int, starts: List[np.ndarray], max_scale: float) -> Optional[np.ndarray]:
    """Run SLSQP on the unit simplex from several starts; return the best point."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return None

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(C)]
    best, best_val = None, np.inf
    for w0 in starts:
        try:
            res = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
        except Exception:
            continue
        w = np.asarray(res.x, dtype=np.float64)
        if not np.isfinite(w).all() or not np.isclose(w.sum(), 1.0, atol=1e-6):
            continue
        val = float(objective(w))
        if val < best_val - 1e-18:
            best, best_val = w, val
    _ = max_scale
    return best


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

    # Long-only GMV via quadratic programming.  The objective is rescaled so its
    # magnitude is O(1); otherwise SLSQP's relative termination test passes at
    # the initial point and returns 1/N.
    scale = float(np.trace(cov)) / C if C else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    cov_scaled = cov / scale

    def objective(w):
        return float(w @ cov_scaled @ w)

    starts = _warm_starts(cov, C)
    best = _minimise_simplex(objective, C, starts, scale)
    if best is None:
        warnings.warn(
            "long-only GMV optimisation unavailable (scipy missing or failed); "
            "falling back to inverse-variance weights",
            RuntimeWarning,
            stacklevel=2,
        )
        return _warm_starts(cov, C)[1]
    if objective(best) >= objective(starts[0]) - 1e-18:
        warnings.warn(
            "long-only GMV optimiser did not improve on equal weights; "
            "returning inverse-variance weights",
            RuntimeWarning,
            stacklevel=2,
        )
        return _warm_starts(cov, C)[1]
    return best


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

    # Long-only tangency via QP.  Sharpe is scale free, but SLSQP can still
    # terminate at the initial point on ill-conditioned covariances, so try
    # several starts and keep the best.
    vol = np.sqrt(np.clip(np.diag(cov), 1e-18, None))

    def neg_sharpe(w):
        port_ret = float(w @ mean_ret)
        port_vol = float(np.sqrt(max(w @ cov @ w, 1e-18)))
        return -(port_ret - rf) / port_vol

    starts = _warm_starts(cov, C)
    # risk-parity-ish / max-mean starts are informative for tangency
    starts = starts + [np.clip(excess, 0.0, None) / max(np.clip(excess, 0.0, None).sum(), 1e-15)]
    starts = [np.clip(excess / vol, 0.0, None) / max(np.clip(excess / vol, 0.0, None).sum(), 1e-15)] + starts
    starts = [s for s in starts if np.isfinite(s).all() and s.sum() > 0]

    best = _minimise_simplex(neg_sharpe, C, starts, 1.0)
    if best is None:
        warnings.warn(
            "long-only tangency optimisation unavailable (scipy missing or failed); "
            "falling back to long-only excess-return weights",
            RuntimeWarning,
            stacklevel=2,
        )
        w = np.clip(excess / vol, 0.0, None)
        return w / w.sum() if w.sum() > 0 else np.ones(C) / C
    return best


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

        # Add ground-truth comparison: estimate from real data.
        # The key is built explicitly.  It used to be assembled with
        # name_suffix.split('_'), which mapped BOTH "_Real_GMV" and
        # "_Real_GMV_LongOnly" onto "Real_GT_RealGMV" (and likewise for
        # tangency): the long-only variant silently overwrote the unconstrained
        # oracle, so every "real GMV oracle" in the results was in fact the
        # long-only one.
        real_cov = _sample_covariance(real)
        real_mean = real.mean(axis=0)
        real_baselines = [
            ("Real_GT_RealGMV", _minimum_variance_weights, True),
            ("Real_GT_RealGMV_LongOnly", _minimum_variance_weights, False),
            ("Real_GT_RealTangency", _tangency_weights, True),
            ("Real_GT_RealTangency_LongOnly", _tangency_weights, False),
        ]
        for key, fn, use_short in real_baselines:
            if fn is _minimum_variance_weights:
                w_real = fn(real_cov, allow_short=use_short)
            else:
                w_real = fn(real_cov, real_mean, allow_short=use_short, rf=self.rf)
            w_real = w_real / w_real.sum()
            port_returns = real @ w_real
            results[key] = {
                **_portfolio_metrics(port_returns, weights_history=w_real[np.newaxis, :],
                                    periods_per_year=self.periods_per_year),
                "weights": w_real.tolist(),
            }

        return results
