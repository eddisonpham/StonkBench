"""U1–U5 PnL Toolbox (paper §6.1.3, eq. 27).

For a single test window, given a per-period PnL array of shape ``(T,)``,
computes:

- **U1 PnL total**   — sum of period PnL.
- **U2 Sharpe**      — mean / std of period PnL. *No* √L annualisation
  (single trading-year horizon).
- **U3 CVaR(α)**     — expected loss in the worst α-tail of period PnL.
  Returns a negative number (loss) when there is one.
- **U4 Win rate**    — fraction of periods with positive PnL.
- **U5 Max DD**      — largest peak-to-trough decline of the cumulative
  PnL curve.

Aggregation across N windows (and optionally C channels) is reported as
``mean ± std`` for every metric.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


# Display keys — keep these stable for downstream consumers.
METRIC_KEYS = ("pnl", "sharpe", "cvar", "win_rate", "max_drawdown")


CUM_SUM_CLAMP = 40.0  # saturation ceiling for floating-point guard


def safe_exp_cumsum(log_returns: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """``exp(clamp(cumsum(log_returns), -C, C))``.

    Guards float32 overflow when reconstructing prices from large
    log returns. Without the clamp a single adversarial window with
    |r| ≫ 1 raises cumsum past ±90 and ``exp`` returns ``inf`` or
    ``NaN`` downstream. Saturating to ``±C`` yields finite but ceiling'd
    prices — preferable to propagating non-finite values into the
    U1–U5 toolbox.
    """
    c = torch.cumsum(log_returns, dim=dim)
    c = c.clamp(min=-CUM_SUM_CLAMP, max=CUM_SUM_CLAMP)
    return torch.exp(c)


class MetricToolbox:
    """U1–U5 per-window + cross-window aggregation."""

    DEFAULT_ALPHA = 0.05

    # ------------------------------------------------------------------ #
    # Single-window metrics                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute(period_pnl: torch.Tensor, alpha: float = DEFAULT_ALPHA) -> Dict[str, float]:
        """Compute U1–U5 for a single window's per-period PnL.

        Args:
            period_pnl: shape ``(T,)`` per-period PnL in dollars (or
                per initial dollar). T is the number of *trading* periods.
            alpha: CVaR tail level. Defaults to 0.05 (worst 5%).

        Returns:
            Dict with keys ``pnl``, ``sharpe``, ``cvar``, ``win_rate``,
            ``max_drawdown``.
        """
        pnl = period_pnl.detach().cpu().float().reshape(-1)
        T = pnl.shape[0]
        if T == 0:
            return {k: float("nan") for k in METRIC_KEYS}

        u_pnl = float(pnl.sum().item())
        mean_p = float(pnl.mean().item())
        std_p = float(pnl.std(unbiased=False).item()) if T > 1 else 0.0
        u_sharpe = mean_p / std_p if std_p > 1e-12 else 0.0

        # CVaR: mean of the bottom-α tail of per-period PnL.
        k = max(1, int(np.ceil(alpha * T)))
        sorted_pnl, _ = torch.sort(pnl)
        tail = sorted_pnl[:k]
        u_cvar = float(tail.mean().item())

        u_win = float((pnl > 0).to(torch.float32).mean().item())

        cum = torch.cumsum(pnl, dim=0)
        running_max = torch.cummax(cum, dim=0).values
        drawdown = running_max - cum
        u_dd = float(drawdown.max().item())

        return {
            "pnl": u_pnl,
            "sharpe": u_sharpe,
            "cvar": u_cvar,
            "win_rate": u_win,
            "max_drawdown": u_dd,
        }

    # ------------------------------------------------------------------ #
    # Aggregation across windows / channels                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def aggregate(
        window_metrics: Sequence[Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate per-window metrics as mean ± std (eq. 27).

        Args:
            window_metrics: list of dicts, each output of :func:`compute`.
                NaN values are dropped per metric.
        """
        out: Dict[str, Dict[str, float]] = {}
        if not window_metrics:
            return {k: {"mean": float("nan"), "std": float("nan")} for k in METRIC_KEYS}
        for key in METRIC_KEYS:
            vals = np.asarray(
                [m[key] for m in window_metrics if key in m and not np.isnan(m[key])],
                dtype=np.float64,
            )
            if vals.size == 0:
                out[key] = {"mean": float("nan"), "std": float("nan")}
            else:
                out[key] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()) if vals.size > 1 else 0.0,
                }
        return out

    @staticmethod
    def aggregate_per_channel(
        channel_aggregates: Sequence[Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, float]]:
        """Average the per-channel (mean±std) into a single (mean±std).

        Used when the same task is run on C independent channels (Options).
        For each metric the final report is the average of the channel
        means; the standard deviation of channel means is also reported.
        """
        out: Dict[str, Dict[str, float]] = {}
        if not channel_aggregates:
            return {k: {"mean": float("nan"), "std": float("nan")} for k in METRIC_KEYS}
        for key in METRIC_KEYS:
            means = np.asarray(
                [c[key]["mean"] for c in channel_aggregates if not np.isnan(c[key]["mean"])],
                dtype=np.float64,
            )
            if means.size == 0:
                out[key] = {"mean": float("nan"), "std": float("nan")}
            else:
                out[key] = {
                    "mean": float(means.mean()),
                    "std": float(means.std()) if means.size > 1 else 0.0,
                }
        return out


def per_period_pnl_from_positions(
    positions: torch.Tensor,
    prices: torch.Tensor,
    mode: str = "simple",
) -> torch.Tensor:
    """Helper: convert a position/holding tensor into per-period PnL.

    Convention: ``positions[t]`` is the holding chosen at end of step
    ``t``, which earns the return from step ``t`` to step ``t+1``.
    Therefore we use positions indexed ``0..L-2`` against prices
    ``0..L-1``, giving a per-period PnL of shape ``(N, L-1)``.

    Args:
        positions: ``(N, L)`` (last column may be unused for trading PnL)
        prices: ``(N, L)``
        mode: ``"simple"`` uses ``S_t * (exp(r) - 1)``; ``"log"`` uses
            ``S_t * r``. Inputs come in as log returns, so ``r =
            log(S_{t+1}/S_t)``.

    Returns:
        ``(N, L-1)`` per-period PnL.
    """
    if mode not in ("simple", "log"):
        raise ValueError(f"mode must be 'simple' or 'log', got {mode!r}")

    if positions.shape != prices.shape:
        raise ValueError(
            f"positions shape {positions.shape} must match prices shape {prices.shape}"
        )

    # The holding at end of day t acts on the move from S_t to S_{t+1}.
    held = positions[:, :-1]  # (N, L-1) or (N, L-1, C)
    if mode == "simple":
        # S_{t+1} - S_t = S_t * (exp(log_return) - 1)
        # We need log returns; derive from prices:
        log_returns = torch.log(prices[:, 1:] / prices[:, :-1])
        delta_price = prices[:, :-1] * (torch.exp(log_returns) - 1.0)
    else:
        # Use log returns directly as the linear approximation.
        log_returns = torch.log(prices[:, 1:] / prices[:, :-1])
        delta_price = prices[:, :-1] * log_returns

    if prices.ndim == 3:
        # Multi-asset (N, L, C): sum per-asset per-period PnL into a
        # single portfolio PnL (N, L-1).
        return (held * delta_price).sum(dim=-1)
    return held * delta_price  # (N, L-1) for the 2D (single-asset) path
