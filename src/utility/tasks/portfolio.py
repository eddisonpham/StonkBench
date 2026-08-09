"""§6.3 Portfolio hedging task (scaffold).

The whole multi-asset window of shape ``(N, L, I+J)`` is fed to a single
:class:`PortfolioLSTM` (training is on a *single* model, not per-channel).
This file wires the policy to the U1–U5 toolbox.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.utility.tasks.base import BaseUtilityTask
from src.utility.policies.portfolio_lstm import PortfolioLSTM
from src.utility.metrics import per_period_pnl_from_positions, safe_exp_cumsum


class PortfolioTask(BaseUtilityTask):
    task_name = "portfolio"
    policy_family = "lstm"
    is_per_channel = False

    def __init__(
        self,
        num_inventory: int = 20,
        num_hedge: int = 5,
        seq_length: int = 252,
        pnl_returns: str = "simple",
        channel_map: Optional[Dict[str, list]] = None,
        default_initial_dollar: float = 1.0,
    ):
        """``channel_map`` optionally maps roles to channel indices:
        ``{"inventory": [0..I-1], "hedge": [I..I+J-1]}``. Defaults to
        ``{"inventory": list(range(I))}`` and ``{"hedge": list(range(I, I+J))}``
        which is the paper convention.
        ``default_initial_dollar`` is the starting portfolio dollar value
        per window (default $1, consistent with paper's unit-USD reporting).
        """
        super().__init__(pnl_returns=pnl_returns)
        self.I = int(num_inventory)
        self.J = int(num_hedge)
        self.seq_length = seq_length
        self.default_initial_dollar = float(default_initial_dollar)
        if channel_map is None:
            self.channel_map = {
                "inventory": list(range(self.I)),
                "hedge": list(range(self.I, self.I + self.J)),
            }
        else:
            self.channel_map = {k: list(map(int, v)) for k, v in channel_map.items()}
            inv = self.channel_map.get("inventory", [])
            hdg = self.channel_map.get("hedge", [])
            if len(inv) != self.I or len(hdg) != self.J:
                raise ValueError(
                    f"channel_map inventory/hedge sizes {len(inv)}/{len(hdg)} "
                    f"do not match I={self.I}, J={self.J}"
                )

    # ------------------------------------------------------------------ #
    def prepare_training(self, log_returns: torch.Tensor, **_: Any) -> torch.Tensor:
        """Reconstruct prices from log returns; the LSTM consumes prices.
        Channel ordering follows ``self.channel_map``."""
        if log_returns.ndim != 3:
            raise ValueError(
                f"PortfolioTask expects (N, L, I+J) windows (full multi-asset), "
                f"got {tuple(log_returns.shape)}"
            )
        N, L, C = log_returns.shape
        if C != self.I + self.J:
            raise ValueError(
                f"Channel count {C} != I+J = {self.I + self.J}. The §6.3 window "
                f"must hold {self.I} inventory stocks and {self.J} hedge ETFs."
            )
        prices = torch.empty(N, L + 1, C, dtype=torch.float32)
        prices[:, 0, :] = 1.0
        prices[:, 1:, :] = safe_exp_cumsum(log_returns, dim=1)
        return prices

    # ------------------------------------------------------------------ #
    def build_policy(self, **_: Any) -> PortfolioLSTM:
        return PortfolioLSTM(
            seq_length=self.seq_length,
            num_inventory=self.I,
            num_hedge=self.J,
        )

    # ------------------------------------------------------------------ #
    def _assemble_full_pos(
        self,
        inventory_unit: torch.Tensor,
        hedge_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Place inventory + hedge positions into the (N, L, I+J) tensor
        according to ``self.channel_map``. The LSTM's hedge output axis is
        in the same order as ``channel_map["hedge"]``.
        """
        N, L, _ = hedge_positions.shape
        full = torch.zeros(N, L, self.I + self.J, device=hedge_positions.device,
                           dtype=hedge_positions.dtype)
        inv_idx = torch.as_tensor(self.channel_map["inventory"],
                                  device=hedge_positions.device, dtype=torch.long)
        hdg_idx = torch.as_tensor(self.channel_map["hedge"],
                                  device=hedge_positions.device, dtype=torch.long)
        # Static unit inventory per asset (paper §6.3 P_t = 1).
        full[:, :, inv_idx] = inventory_unit.to(hedge_positions.device,
                                                dtype=hedge_positions.dtype)
        full[:, :, hdg_idx] = hedge_positions
        return full

    # ------------------------------------------------------------------ #
    def predict_period_pnl(
        self,
        policy: PortfolioLSTM,
        test_windows: torch.Tensor,
        initial_dollars: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        """Per-window per-period PnL, scaled by ``initial_dollars``.

        Each test window is tagged with its own starting $ amount (per the
        user spec). ``initial_dollars`` is shape ``(N,)``; defaults to
        ``self.default_initial_dollar`` for every window.
        """
        prices = self.prepare_training(test_windows)
        positions = policy.predict(prices)  # (N, L, J) hedge positions (J-axis = channel_map["hedge"])
        inventory_unit = torch.ones((), dtype=torch.float32)
        full_pos = self._assemble_full_pos(inventory_unit, positions)  # (N, L, I+J)
        pnl_unit = per_period_pnl_from_positions(
            full_pos, prices, mode=self.pnl_returns
        )  # (N, L-1)
        N = prices.shape[0]
        if initial_dollars is None:
            init = torch.full((N,), self.default_initial_dollar,
                              dtype=torch.float32, device=prices.device)
        else:
            init = initial_dollars.to(prices.device, dtype=torch.float32).reshape(-1)
            if init.shape[0] == 1:
                init = init.expand(N).contiguous()
            elif init.shape[0] != N:
                raise ValueError(
                    f"initial_dollars has length {init.shape[0]} but there "
                    f"are {N} test windows — pass shape ({N},), shape "
                    f"(1,) for a shared amount, or None for the default "
                    f"(${self.default_initial_dollar:g})."
                )
        return pnl_unit * init.unsqueeze(-1)  # (N, L-1) in dollars

    # ------------------------------------------------------------------ #
    def extras(
        self, policy: PortfolioLSTM, test_windows: torch.Tensor, **_: Any
    ) -> Dict[str, float]:
        with torch.no_grad():
            positions = policy.predict(self.prepare_training(test_windows))
            inventory_total = float(self.I)
            net = inventory_total + positions.sum(dim=2)  # (N, L)
            return {
                "net_exposure_mean_abs": float(net.abs().mean().item()),
                "net_exposure_max_abs": float(net.abs().max().item()),
            }

    # ------------------------------------------------------------------ #
    def run_one(
        self,
        train_windows: torch.Tensor,
        test_windows: torch.Tensor,
        initial_dollars: Optional[torch.Tensor] = None,
        *,
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        verbose: bool = False,
    ):
        prepared = self.prepare_training(train_windows)
        policy = self.build_policy()
        policy.fit(
            prepared, num_epochs=num_epochs,
            batch_size=batch_size, learning_rate=learning_rate,
            verbose=verbose,
        )
        pnl = self.predict_period_pnl(policy, test_windows, initial_dollars=initial_dollars)
        ex = self.extras(policy, test_windows)
        return policy, pnl, ex
