"""§6.4 Alpha generation task (scaffold).

Whole multi-asset window of shape ``(N, L, I=25)`` is fed to a single
:class:`AlphaLSTM`. The loss is the negative differentiable Sharpe ratio.
The per-period PnL uses the policy's softmax weights directly.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from src.utility.tasks.base import BaseUtilityTask
from src.utility.policies.alpha_lstm import AlphaLSTM
from src.utility.metrics import per_period_pnl_from_positions, safe_exp_cumsum


class AlphaTask(BaseUtilityTask):
    task_name = "alpha"
    policy_family = "lstm"
    is_per_channel = False

    def __init__(
        self,
        num_assets: int = 25,
        seq_length: int = 252,
        pnl_returns: str = "simple",
        long_only: bool = True,
        default_initial_dollar: float = 1.0,
    ):
        """``long_only=True`` is the paper default — softmax weights that
        sum to 1 per step. ``long_only=False`` allows long-short in
        [-1, +1]. ``default_initial_dollar`` is the start-of-window dollar
        value (per-window scaling applied at evaluation time)."""
        super().__init__(pnl_returns=pnl_returns)
        self.I = int(num_assets)
        self.seq_length = seq_length
        self.long_only = bool(long_only)
        self.default_initial_dollar = float(default_initial_dollar)

    # ------------------------------------------------------------------ #
    def prepare_training(self, log_returns: torch.Tensor, **_: Any) -> torch.Tensor:
        if log_returns.ndim != 3:
            raise ValueError(
                f"AlphaTask expects (N, L, I) windows, got {tuple(log_returns.shape)}"
            )
        N, L, C = log_returns.shape
        if C != self.I:
            raise ValueError(f"Channel count {C} != I = {self.I}")
        prices = torch.empty(N, L + 1, C, dtype=torch.float32)
        prices[:, 0, :] = 1.0
        prices[:, 1:, :] = safe_exp_cumsum(log_returns, dim=1)
        return prices

    # ------------------------------------------------------------------ #
    def build_policy(self, **_: Any) -> AlphaLSTM:
        return AlphaLSTM(seq_length=self.seq_length, num_assets=self.I,
                          long_only=self.long_only)

    # ------------------------------------------------------------------ #
    def predict_period_pnl(
        self,
        policy: AlphaLSTM,
        test_windows: torch.Tensor,
        initial_dollars: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        prices = self.prepare_training(test_windows)
        weights = policy.predict(prices)  # (N, L, I)
        pnl_unit = per_period_pnl_from_positions(
            weights, prices, mode=self.pnl_returns
        )  # (N, L-1)  -- unit-dollar amounts (weights 0..1)
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
        return pnl_unit * init.unsqueeze(-1)

    # ------------------------------------------------------------------ #
    def extras(
        self, policy: AlphaLSTM, test_windows: torch.Tensor, **_: Any
    ) -> Dict[str, float]:
        with torch.no_grad():
            prices = self.prepare_training(test_windows)
            weights = policy.predict(prices)  # (N, L, I)
            log_r = torch.log(prices[:, 1:] / prices[:, :-1])
            simple_r = torch.exp(log_r) - 1.0
            port_r = (weights[:, :-1] * simple_r).sum(dim=-1)
            mu = port_r.mean(dim=1)
            sd = port_r.std(dim=1, unbiased=False).clamp(min=1e-6)
            sr = (mu / sd).mean()
        return {"sharpe_train_test": float(sr.item())}

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
