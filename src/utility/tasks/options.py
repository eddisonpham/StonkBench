"""§6.2 Options delta hedging task.

Per-channel: an input tensor of shape ``(N, L, C)`` is treated as
``C`` independent option hedging problems. For each channel ``c`` we:

1. Apply 7-moneyness augmentation (each path expanded to 7 sub-examples
   with distinct strikes ``K_j = S_0 / g̃(0)_j``);
2. Compute a Black–Scholes–Merton premium ``c_0(K_j, σ, T)`` per sub-example
   (``σ`` annualised from the training window);
3. Train a fresh :class:`MoneynessLSTM` on the augmented (N*7, L) tensor;
4. Evaluate on test windows at an **ATM strike** ``K_test = mean(S_0)``;
5. Produce per-period PnL for the U1–U5 toolbox.

The produced positions are Δ-trajectories of shape ``(N, L)`` (with the
last column unused for trading PnL, kept for full-window reporting).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utility.tasks.base import BaseUtilityTask
from src.utility.policies.lstm_moneyness import MoneynessLSTM, bs_call_price
from src.utility.metrics import per_period_pnl_from_positions, safe_exp_cumsum


# 7-moneyness grid (paper §6.2). Evenly spaced on {0.7..1.3}.
DEFAULT_MONEYNESS_GRID: Tuple[float, ...] = (
    0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30,
)


class OptionsTask(BaseUtilityTask):
    task_name = "options"
    policy_family = "lstm"
    is_per_channel = True

    def __init__(
        self,
        moneyness_grid: Tuple[float, ...] = DEFAULT_MONEYNESS_GRID,
        premium_mode: str = "bs",
        learnable_premium: bool = False,
        seq_length: int = 252,
        time_horizon_years: float = 1.0,
        annualisation_factor: int = 252,
        pnl_returns: str = "simple",
        default_initial_dollar: float = 1.0,
    ):
        super().__init__(pnl_returns=pnl_returns)
        if premium_mode not in ("bs", "learnable", "zero"):
            raise ValueError(f"premium_mode must be bs|learnable|zero, got {premium_mode!r}")
        self.moneyness_grid = tuple(float(m) for m in moneyness_grid)
        self.premium_mode = premium_mode
        self.learnable_premium = learnable_premium or premium_mode == "learnable"
        self.seq_length = seq_length
        self.T_years = float(time_horizon_years)
        self.annualisation = int(annualisation_factor)
        self.default_initial_dollar = float(default_initial_dollar)

    # ------------------------------------------------------------------ #
    def prepare_training(
        self, log_returns: torch.Tensor, **_: Any
    ) -> Dict[str, torch.Tensor]:
        """Augment training paths with 7 moneyness strikes.

        Args:
            log_returns: ``(N, L)`` log returns for one *channel* of the
                multi-channel training set.

        Returns:
            ``{"prices": (N*7, L+1), "K": (N*7,), "c0": (N*7,)}``.
        """
        if log_returns.ndim != 2:
            raise ValueError(
                f"OptionsTask.prepare_training expects (N, L) for one channel; "
                f"got {tuple(log_returns.shape)}"
            )
        log_returns = log_returns.float()
        N, L_minus_1 = log_returns.shape
        L = L_minus_1  # log_returns convention: each row is L-1 returns
        # Reconstruct prices from a unit-start convention for augmentation.
        # S_0 is taken as 1.0 for each input path; K_j = 1 / g̃(0)_j.
        prices_unit = torch.empty(N, L + 1, dtype=torch.float32)
        prices_unit[:, 0] = 1.0
        prices_unit[:, 1:] = safe_exp_cumsum(log_returns, dim=1)
        S0 = prices_unit[:, 0].clone()  # all 1.0 in this convention

        # σ: annualised realised vol from the training window.
        with torch.no_grad():
            sigma = float(log_returns.std(unbiased=False).item()) * math.sqrt(self.annualisation)

        rows_prices, rows_K, rows_c0 = [], [], []
        for m in self.moneyness_grid:
            K_j = (S0 / m).float()  # (N,)
            c0_j = bs_call_price(
                S0=S0, K=K_j, sigma=sigma, T=self.T_years, r=0.0
            )  # (N,)
            rows_prices.append(prices_unit)
            rows_K.append(K_j)
            rows_c0.append(c0_j)

        prices_aug = torch.cat(rows_prices, dim=0)  # (N*7, L+1)
        K_aug = torch.cat(rows_K, dim=0)
        c0_aug = torch.cat(rows_c0, dim=0)
        return {"prices": prices_aug, "K": K_aug, "c0": c0_aug}

    # ------------------------------------------------------------------ #
    def _reconstruct_prices(
        self, log_returns: torch.Tensor, K_test: torch.Tensor
    ) -> torch.Tensor:
        """Build a (N, L+1) price tensor from (N, L) log returns, with
        ``S_0 = K_test`` so that for an ATM strike ``g̃(0) = 1``."""
        prices = torch.empty(
            log_returns.shape[0],
            log_returns.shape[1] + 1,
            dtype=torch.float32,
        )
        prices[:, 0] = K_test.float()
        prices[:, 1:] = K_test.unsqueeze(-1).float() * safe_exp_cumsum(
            log_returns, dim=1
        )
        return prices

    # ------------------------------------------------------------------ #
    def build_policy(self, **_: Any) -> MoneynessLSTM:
        return MoneynessLSTM(
            seq_length=self.seq_length,
            hidden_size=64,
            num_layers=1,
            premium_mode=self.premium_mode,
            learnable_premium=self.learnable_premium,
        )

    # ------------------------------------------------------------------ #
    def predict_period_pnl(
        self,
        policy: MoneynessLSTM,
        test_log_returns: torch.Tensor,
        initial_dollars: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        """Compute per-window per-period PnL on a single test channel.

        Args:
            test_log_returns: ``(N, L)`` log returns for one channel.
            initial_dollars: optional ``(N,)`` per-window dollar anchor.
                PnL is scaled linearly by ``initial_dollars`` so the
                U1–U5 toolbox reports dollar PnL per window (default
                uses ``self.default_initial_dollar`` for every window).
        Returns:
            ``(N, L-1)`` per-period PnL (toggleable simple/log returns).
        """
        # Choose ATM strike: K_test = mean of reconstructed S_0 (==1 in our
        # unit-S_0 convention; here we make K_test the *mean of S_0 over
        # the channel*).
        N, L = test_log_returns.shape
        # Use unit S_0 convention so that K_test = 1.0 (ATM-equivalent for
        # each window). This keeps the moneyness LSTM architecture trivially
        # consistent with training (S_0 = 1 there too).
        K_test = torch.ones(N, dtype=torch.float32)
        prices = self._reconstruct_prices(test_log_returns, K_test)

        # Get Δ trajectory from the policy (L deltas, same length as prices).
        positions = policy.predict(prices, K=K_test)  # (N, L) == (N, prices.shape[1])

        # Per-period PnL: helper handles the L-1 slicing internally.
        pnl_unit = per_period_pnl_from_positions(
            positions, prices, mode=self.pnl_returns
        )
        if initial_dollars is None:
            init = torch.full(
                (N,), self.default_initial_dollar,
                dtype=torch.float32, device=pnl_unit.device,
            )
        else:
            init = initial_dollars.to(pnl_unit.device, dtype=torch.float32).reshape(-1)
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
        self,
        policy: MoneynessLSTM,
        test_log_returns: torch.Tensor,
        **_: Any,
    ) -> Dict[str, float]:
        """Hedging-loss diagnostic (paper §6.2.4): MSE of replication
        error in moneyness units on the test channel."""
        N, L = test_log_returns.shape
        K_test = torch.ones(N, dtype=torch.float32)
        prices = self._reconstruct_prices(test_log_returns, K_test)
        with torch.no_grad():
            positions = policy.predict(prices, K=K_test)
            # Estimate test σ for the BS premium.
            sigma = float(test_log_returns.std(unbiased=False).item()) * math.sqrt(self.annualisation)
            c0 = bs_call_price(S0=K_test, K=K_test, sigma=sigma, T=self.T_years)
            loss = policy.loss(prices, K_test, c0).item()
        g_terminal = torch.clamp(prices[:, -1] / K_test - 1.0, min=0.0)
        c0_per = c0 / K_test
        g_diff = (prices[:, 1:] / K_test.unsqueeze(-1) - prices[:, :-1] / K_test.unsqueeze(-1))
        residual = g_terminal - c0_per - (positions[:, :-1] * g_diff).sum(dim=1)
        return {
            "hedging_loss_moneyness": float(loss),
            "residual_mean": float(residual.mean().item()),
            "residual_std": float(residual.std().item()),
        }

    # ------------------------------------------------------------------ #
    # Convenience: per-channel pipeline driver                            #
    # ------------------------------------------------------------------ #
    def evaluate_per_channel(
        self,
        policy: MoneynessLSTM,
        test_log_returns_all_channels: torch.Tensor,
        initial_dollars: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        out = []
        for c in range(test_log_returns_all_channels.shape[-1]):
            out.append(self.predict_period_pnl(
                policy, test_log_returns_all_channels[:, :, c],
                initial_dollars=initial_dollars,
            ))
        return out

    # ------------------------------------------------------------------ #
    def run_one_channel(
        self,
        train_log_returns_channel: torch.Tensor,
        test_log_returns_channel: torch.Tensor,
        initial_dollars: Optional[torch.Tensor] = None,
        *,
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        verbose: bool = False,
    ) -> Tuple[Any, torch.Tensor, Dict[str, float]]:
        """Train on one channel and return ``(policy, period_pnl, extras)``."""
        prepared = self.prepare_training(train_log_returns_channel)
        policy = self.build_policy()
        policy.fit(
            prepared, num_epochs=num_epochs,
            batch_size=batch_size, learning_rate=learning_rate,
            verbose=verbose,
        )
        pnl = self.predict_period_pnl(
            policy, test_log_returns_channel,
            initial_dollars=initial_dollars,
        )
        ex = self.extras(policy, test_log_returns_channel)
        return policy, pnl, ex
