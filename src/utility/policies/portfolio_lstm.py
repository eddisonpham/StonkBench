"""Paper §6.3 Portfolio Hedging policy.

Static unit inventory ``P_t = 1`` over ``I=20`` inventory stocks is hedged
by ``J=5`` liquid ETFs. At each step the LSTM picks the hedging position
``π_t ∈ ℝᴶ`` over the J hedge instruments. Loss is the cumulative
absolute net exposure:

::

    E_t = Σ_{i ∈ I} P_t[i] + Σ_{j ∈ J} π_t[j]         (signed inventory)
    L   = Σ_t |E_t|

Inputs are the *whole* multi-asset window of shape ``(N, L, I+J)``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.utility.policies.base import BaseUtilityPolicy
from src.utils.device import get_device


class PortfolioLSTM(nn.Module, BaseUtilityPolicy):
    task_name = "portfolio"
    family = "lstm"

    def __init__(
        self,
        seq_length: int,
        num_inventory: int = 20,
        num_hedge: int = 5,
        hidden_size: int = 64,
        num_layers: int = 1,
        hedge_clip: float | None = 10.0,
        inventory_per_asset: float = 1.0,
    ):
        """``inventory_per_asset`` is the static inventory level P_{t,i}
        applied to **every** inventory asset (paper sets ``P_t = 1`` per
        asset). ``hedge_clip`` optionally clamps the LSTM's hedge output
        ``π_{t,j}`` into ``[-hedge_clip, +hedge_clip]`` to keep the
        positions numerically sane."""
        nn.Module.__init__(self)
        BaseUtilityPolicy.__init__(self)
        self.seq_length = seq_length
        self.I = int(num_inventory)
        self.J = int(num_hedge)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hedge_clip = float(hedge_clip) if hedge_clip is not None else None
        self.inventory_per_asset = float(inventory_per_asset)

        self.lstm = nn.LSTM(
            input_size=self.I + self.J,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, self.J)
        self.device = get_device()
        self.to(self.device)

    # ------------------------------------------------------------------ #
    def predict(self, test_windows: torch.Tensor, **_: Any) -> torch.Tensor:
        """Return hedge positions ``π_t ∈ ℝᴶ`` of shape ``(N, L, J)``.

        ``test_windows`` is the *whole* multi-asset window
        ``(N, L, I+J)``: the first I channels are static inventory and the
        last J are hedge candidates (price-only inputs).

        The LSTM takes the *prices* up to time ``L-1`` and emits one hedge
        position per remaining step ``t = 0..L-1``; the final column is
        zero because there is no return to consume after the maturity.
        """
        self.eval()
        windows = test_windows.to(self.device).float()
        # Use price levels (not returns) for the LSTM; tasks can pre-process.
        out, _ = self.lstm(windows[:, :-1, :])
        pi = self.head(out)  # (N, L-1, J)
        if self.hedge_clip is not None:
            pi = pi.clamp(min=-self.hedge_clip, max=self.hedge_clip)
        # Pad last step with zero so shape is (N, L, J).
        pad = torch.zeros(windows.shape[0], 1, self.J, device=self.device)
        return torch.cat([pi, pad], dim=1)

    # ------------------------------------------------------------------ #
    def loss(self, windows_batch: torch.Tensor) -> torch.Tensor:
        """Paper §6.3 loss — ``L = Σ_t |E_t|`` averaged over the batch.

        Net exposure ``E_t = Σ_{i∈I} P_{t,i} + Σ_{j∈J} π_{t,j}`` with
        static unit inventory ``P_{t,i} = self.inventory_per_asset``.
        """
        pi = self.predict(windows_batch)  # (N, L, J)
        inventory_total = self.I * self.inventory_per_asset
        hedge_total = pi.sum(dim=2)  # (N, L)
        net_exposure = inventory_total + hedge_total  # (N, L)
        per_window = net_exposure.abs().sum(dim=1)  # (N,)
        loss = per_window.mean()
        # NaN/Inf guard that zeros bad batches rather than blowing the
        # optimiser up to 1e6 - safer for training stability.
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
    ) -> "PortfolioLSTM":
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
                print(f"[PortfolioLSTM] epoch {epoch + 1}/{num_epochs}  loss={last_loss:.6f}")
        return self
