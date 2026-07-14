"""Fecamp et al. (2021) — Deep learning for discrete-time hedging in incomplete markets.

Per-asset and multi-asset portfolio hedger architectures. All losses are
differentiable and dispatch through :mod:`src.hedging_models.losses`.

Architecture: feed-forward NN (default 3 layers, 128 hidden, ReLU). Outputs
``seq_length - 1`` action deltas; one per rebalancing step. Bounded mode
applies sigmoid to constrain portfolio weights to ``[0, 1]``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.hedging_models.losses import cvar_loss, entropic_risk_loss, log_utility_loss


VALID_LOSS_TYPES = ("cvar", "entropic", "log_utility", "mse")
VALID_ACTIVATIONS = ("relu", "tanh", "gelu")


class FecampHedger(nn.Module):
    """Per-asset Fecamp hedger: NN outputs ``seq_length - 1`` deltas from a window."""

    def __init__(
        self,
        seq_length: int,
        input_dim: int = 1,
        hidden_dim: int = 128,
        n_layers: int = 3,
        activation: str = "relu",
        loss_type: str = "cvar",
        loss_alpha: float = 0.05,
        loss_lambda: float = 1.0,
        bounded: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if seq_length < 2:
            raise ValueError(f"seq_length must be >= 2 (got {seq_length})")
        if loss_type not in VALID_LOSS_TYPES:
            raise ValueError(f"loss_type must be in {VALID_LOSS_TYPES} (got {loss_type!r})")
        if activation not in VALID_ACTIVATIONS:
            raise ValueError(f"activation must be in {VALID_ACTIVATIONS} (got {activation!r})")
        self.seq_length = int(seq_length)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.action_dim = self.seq_length - 1
        self.loss_type = loss_type
        self.loss_alpha = float(loss_alpha)
        self.loss_lambda = float(loss_lambda)
        self.bounded = bool(bounded)

        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        layers: list[nn.Module] = []
        in_dim = self.seq_length * self.input_dim
        for _ in range(self.n_layers):
            layers += [nn.Linear(in_dim, self.hidden_dim), act_cls(), nn.Dropout(float(dropout))]
            in_dim = self.hidden_dim
        layers += [nn.Linear(in_dim, self.action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        if prices.ndim == 3:
            prices = prices.reshape(prices.shape[0], -1)
        out = self.net(prices)
        if self.bounded:
            out = torch.sigmoid(out)
        return out

    def compute_pnl(
        self,
        log_returns: torch.Tensor,
        deltas: torch.Tensor,
        transaction_cost: float = 0.0,
    ) -> torch.Tensor:
        """Hedger PnL = sum_t delta[t] * (S[t+1] - S[t]) - transaction cost.

        Supports both per-asset (log_returns (N, L), deltas (N, L-1) -> PnL (N,))
        and multi-asset (log_returns (N, L, C), deltas (N, L-1) -> PnL (N,)) where
        deltas are an intentionally scalar-per-step hedge ratio (one tradable
        instrument across multiple assets). For a true per-asset-per-step delta
        matrix (N, L-1, C), subclass and override.

        Multi-asset broadcast: deltas (N, L-1) -> (N, L-1, 1) before elementwise
        multiplication with price_path[:, 1:] (N, L-1, C). Without that unsqueeze,
        PyTorch broadcasts on the trailing dim and the dim-1 sizes 9 vs 9 vs 2
        produce a ``RuntimeError`` at non-singleton dim 1.
        """
        price_path = log_returns.cumsum(dim=1)
        if price_path.ndim == 3 and deltas.ndim == 2:
            deltas_eff = deltas.unsqueeze(-1)  # (N, L-1, 1)
            reduce_dims: tuple[int, ...] = (1, 2)
        elif price_path.ndim == deltas.ndim:
            deltas_eff = deltas
            reduce_dims = (1,)
        else:
            raise ValueError(
                f"deltas shape {tuple(deltas.shape)} cannot broadcast with "
                f"price_path shape {tuple(price_path.shape)}"
            )
        gross = (deltas_eff * price_path[:, 1:]).sum(dim=reduce_dims)
        if transaction_cost > 0.0:
            delta_changes = torch.zeros_like(deltas)
            delta_changes[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
            cost = transaction_cost * delta_changes.abs().sum(dim=1)
        else:
            cost = torch.zeros(log_returns.shape[0], device=log_returns.device)
        return gross - cost

    def compute_loss(
        self,
        log_returns: torch.Tensor,
        deltas: Optional[torch.Tensor] = None,
        transaction_cost: float = 0.0,
    ) -> torch.Tensor:
        if deltas is None:
            deltas = self.forward(log_returns)
        pnl = self.compute_pnl(log_returns, deltas, transaction_cost=transaction_cost)
        if self.loss_type == "cvar":
            return cvar_loss(pnl, alpha=self.loss_alpha)
        if self.loss_type == "entropic":
            return entropic_risk_loss(pnl, lambda_coef=self.loss_lambda)
        if self.loss_type == "log_utility":
            return log_utility_loss(pnl)
        return (pnl ** 2).mean()  # MSE fallback


class FecampPortfolioHedger(nn.Module):
    """Multi-asset portfolio hedger: per-asset FecampHedger + log-utility loss."""

    def __init__(
        self,
        seq_length: int,
        n_assets: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        loss_type: str = "log_utility",
        transaction_cost: float = 0.0,
    ) -> None:
        super().__init__()
        if loss_type not in ("log_utility",):
            raise ValueError("Portfolio hedger only supports log_utility loss")
        self.hedger = FecampHedger(
            seq_length=seq_length,
            input_dim=n_assets,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            loss_type=loss_type,
            bounded=False,
        )
        self.transaction_cost = float(transaction_cost)

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        return self.hedger(returns)

    def compute_loss(self, returns: torch.Tensor) -> torch.Tensor:
        deltas = self.hedger(returns)
        pnl = self.hedger.compute_pnl(returns, deltas, transaction_cost=self.transaction_cost)
        # compute_pnl with (N, L, C) input broadcasts the scalar delta across C assets
        # then reduces over time → (N, C). Sum across assets first to get a per-sample
        # portfolio PnL (N,) before applying log-utility, otherwise we'd apply the utility
        # per-asset independently rather than over the portfolio's true PnL.
        if pnl.ndim > 1:
            pnl = pnl.sum(dim=-1)
        return log_utility_loss(pnl)

    def compute_pnl(
        self,
        returns: torch.Tensor,
        deltas: torch.Tensor,
        transaction_cost: Optional[float] = None,
    ) -> torch.Tensor:
        """Pass-through to the wrapped FecampHedger.compute_pnl.

        Required because callers that abstract over per-asset and portfolio hedgers
        (notably FHE.evaluate()) call ``hedger.compute_pnl(...)`` uniformly. Without
        this method, portfolio mode raises
        ``AttributeError: FecampPortfolioHedger has no attribute 'compute_pnl'``.

        Cost semantics match ``compute_loss``:
          - ``transaction_cost`` is ``None`` (default)  ➜ fall back to ``self.transaction_cost`` (constructor-stored).
          - ``transaction_cost`` is a ``float``          ➜ override the stored value per-call.
        """
        effective_cost = self.transaction_cost if transaction_cost is None else transaction_cost
        return self.hedger.compute_pnl(returns, deltas, transaction_cost=effective_cost)
