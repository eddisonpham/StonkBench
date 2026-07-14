"""Fecamp-style differentiable hedging-risk losses.

* :func:`cvar_loss` — Conditional Value-at-Risk at ``alpha``-quantile. Detached
  quantile + soft mask ensures backprop flows through the tail-mean only.
* :func:`entropic_risk_loss` — ``E[exp(-lambda * PnL)]`` (entropic utility).
* :func:`log_utility_loss` — concave :math:`\\log(1 + PnL)`, portfolio default.

The vendor deep hedgers under ``src/hedging_models/deep_hedgers/`` use MSE;
this module provides Fecamp-aligned alternatives invoked by
``FecampHedger.compute_loss``.
"""

from __future__ import annotations

import torch


def cvar_loss(pnl: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """CVaR (lower-tail expected shortfall) at the ``alpha``-quantile.

    Differentiable approximation: detached ``torch.quantile`` + soft mask so
    the backward path only updates through the tail-mean. ``alpha=0.05`` is the
    Fecamp-paper default.
    """
    if pnl.numel() == 0:
        return torch.zeros((), device=pnl.device)
    var = torch.quantile(pnl, alpha).detach()
    tail_mask = (pnl <= var).float().detach()
    n_tail = tail_mask.sum().clamp(min=1.0)
    tail_mean = (pnl * tail_mask).sum() / n_tail
    return -tail_mean  # minimize -tail_mean = maximize tail


def entropic_risk_loss(pnl: torch.Tensor, lambda_coef: float = 1.0) -> torch.Tensor:
    """Entropic risk measure ``E[exp(-lambda * PnL)]``. Lower = better."""
    return torch.exp(-lambda_coef * pnl).mean()


def log_utility_loss(pnl: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Concave utility maximization: ``-E[log(1 + PnL)]``.

    Multi-asset portfolio PnL is summed per asset before this loss is applied
    (caller responsibility).
    """
    safe = torch.clamp(pnl, min=-1.0 + eps)
    return -torch.log1p(safe).mean()
