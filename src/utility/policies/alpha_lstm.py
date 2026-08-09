"""Paper §6.4 Alpha Generation policy.

Cross-sectional LSTM: at each step the LSTM outputs portfolio weights
``π_t ∈ ℝᴵ`` over ``I=25`` assets (20 stocks + 5 ETFs). Loss is the
**negative differentiable Sharpe** of the realised per-period PnL over
the window:

::

    r̂_t = π_{t-1} · r_t                  (per-period portfolio return)
    μ   = mean(r̂) ;   σ   = std(r̂)
    SR  = μ / σ                          (differentiable Sharpe)
    L   = - SR                           (we *minimise* this)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utility.policies.base import BaseUtilityPolicy
from src.utils.device import get_device


class AlphaLSTM(nn.Module, BaseUtilityPolicy):
    task_name = "alpha"
    family = "lstm"

    def __init__(
        self,
        seq_length: int,
        num_assets: int = 25,
        hidden_size: int = 64,
        num_layers: int = 1,
        long_only: bool = True,
        sharpe_eps: float = 1e-4,
    ):
        """``long_only=True`` (paper-default) applies a cross-sectional
        softmax so the per-step weights sum to 1 (long-only fully invested
        portfolio). ``long_only=False`` uses ``tanh`` so the weights are
        bounded in ``[-1, +1]`` per asset (long-short).

        ``sharpe_eps`` is the floor on the per-window standard deviation in
        the differentiable Sharpe ratio to avoid division-by-zero and
        exploding gradients."""
        nn.Module.__init__(self)
        BaseUtilityPolicy.__init__(self)
        self.seq_length = seq_length
        self.I = int(num_assets)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.long_only = bool(long_only)
        self.sharpe_eps = float(sharpe_eps)

        self.lstm = nn.LSTM(
            input_size=self.I, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.head = nn.Linear(hidden_size, self.I)
        self.device = get_device()
        self.to(self.device)

    # ------------------------------------------------------------------ #
    def predict(self, test_windows: torch.Tensor, **_: Any) -> torch.Tensor:
        """Return per-step weights ``π_t ∈ ℝᴵ`` of shape ``(N, L, I)``."""
        self.eval()
        windows = test_windows.to(self.device).float()
        out, _ = self.lstm(windows[:, :-1, :])
        logits = self.head(out)
        # Cross-sectional constraint: softmax (long-only) or tanh
        # (long-short, bounded in [-1, +1]).
        if self.long_only:
            weights = F.softmax(logits, dim=-1)  # (N, L-1, I), sums to 1
        else:
            weights = torch.tanh(logits)  # (N, L-1, I), in [-1, +1]
        pad = torch.zeros(windows.shape[0], 1, self.I, device=self.device)
        if self.long_only:
            pad[:, 0, :] = 1.0 / self.I
        return torch.cat([weights, pad], dim=1)

    # ------------------------------------------------------------------ #
    def loss(self, windows_batch: torch.Tensor) -> torch.Tensor:
        """Paper §6.4 — *negative differentiable Sharpe* on per-window
        portfolio returns, averaged across the batch."""
        weights = self.predict(windows_batch)  # (N, L, I)
        # Returns from windowed prices: from log-returns (caller passes
        # prices reconstructed with S_0 = 1), compute simple returns for
        # the dollar PnL convention used by the U1–U5 toolbox.
        log_r = torch.log(
            windows_batch[:, 1:, :] / windows_batch[:, :-1, :]
        )  # (N, L-1, I)
        simple_r = torch.exp(log_r) - 1.0
        # Position from t-1 acts on return at t -> shift one step.
        held = weights[:, :-1, :]  # (N, L-1, I)
        port_r = (held * simple_r).sum(dim=-1)  # (N, L-1)
        mu = port_r.mean(dim=1)
        sigma = port_r.std(dim=1, unbiased=False).clamp(min=self.sharpe_eps)
        # Guard against degenerate windows where μ≈0 and σ≈eps so
        # |μ/σ| cannot dominate the loss.
        mu_safe = torch.where(
            sigma.abs() < self.sharpe_eps * 10,
            torch.zeros_like(mu),
            mu,
        )
        sharpe = (mu_safe / sigma).mean()
        loss = -sharpe
        return torch.where(
            torch.isnan(loss) | torch.isinf(loss),
            torch.zeros_like(loss),
            loss,
        )

    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_data: torch.Tensor,
        *,
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        val_data: Optional[torch.Tensor] = None,
        extra: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> "AlphaLSTM":
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        n = train_data.shape[0]
        last_loss = float("nan")
        for epoch in range(num_epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            nb = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = self.loss(train_data[idx])
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
            last_loss = total / max(1, nb)
            if verbose and (epoch % max(1, num_epochs // 5) == 0):
                print(f"[AlphaLSTM] epoch {epoch + 1}/{num_epochs}  loss={last_loss:.6f}")
        return self
