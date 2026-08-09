"""Static Black–Scholes delta baseline (paper-allowed baseline for Options).

Per the user spec, this baseline sets the **premium to zero**, so it is
not actually buying an option: it is a pure static-delta tracking
portfolio, intended as a non-learned reference. It still uses the BS
delta formula with ``r=0`` and a fixed annualised vol estimate, and is
treated as just another ``BaseUtilityPolicy`` so the same evaluator
machinery (TSTR / Augmented protocols, U1–U5 toolbox) can score it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

from src.utility.policies.lstm_moneyness import _norm_cdf
from src.utility.policies.base import BaseUtilityPolicy
from src.utility.metrics import safe_exp_cumsum


class BSStaticDelta(BaseUtilityPolicy):
    task_name = "options"
    family = "bs_static"

    def __init__(
        self,
        seq_length: int,
        K: float = 1.0,
        sigma_annual: float = 0.2,
        time_horizon_years: float = 1.0,
    ):
        """``K`` is the strike; ``sigma_annual`` is the BS-vol estimate;
        ``time_horizon_years`` is the BS time-to-maturity in years (the
        paper's L days ≈ 1 trading year)."""
        self.seq_length = seq_length
        self.K = float(K)
        self.sigma = float(sigma_annual)
        self.T_years = float(time_horizon_years)

    # ------------------------------------------------------------------ #
    def predict(
        self,
        test_paths: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        sigma_annual: Optional[float] = None,
        **_: Any,
    ) -> torch.Tensor:
        """BS delta for each time-step. ``test_paths`` may be prices
        ``(N, L)`` *or* log returns ``(N, L-1)`` — if it has L-1 columns
        we reconstruct prices using ``S_0 = K``."""
        paths = test_paths.float()
        N, L = paths.shape
        if K is None:
            K = torch.full((N,), self.K, dtype=torch.float32)
        K = K.to(paths.device).float().reshape(-1)
        if K.shape[0] == 1:
            K = K.expand(N)
        elif K.shape[0] != N:
            raise ValueError(
                f"BSStaticDelta.predict: K has length {K.shape[0]} but there "
                f"are {N} paths — pass a scalar, shape ({N},), or None for the "
                f"constructor's K={self.K}."
            )
        if sigma_annual is None:
            sigma_annual = self.sigma
        if sigma_annual <= 0:
            return torch.zeros(N, L, device=paths.device)

        if L == self.seq_length - 1:
            # log returns: reconstruct prices from K (per path)
            prices = torch.empty(N, L + 1, device=paths.device, dtype=torch.float32)
            prices[:, 0] = K
            prices[:, 1:] = K.unsqueeze(-1) * safe_exp_cumsum(paths, dim=1)
        else:
            prices = paths

        # Time-to-maturity per step, uniform across paths.
        T = prices.shape[1] - 1
        tau = torch.linspace(self.T_years, 0.0, T + 1, device=paths.device)
        deltas = []
        for t in range(T + 1):
            d = bs_delta_step(prices[:, t], K, sigma_annual, tau[t])
            deltas.append(d)
        return torch.stack(deltas, dim=1)  # (N, L)

    # ------------------------------------------------------------------ #
    def fit(self, *args, **kwargs):
        # Static, no training.
        return self


def bs_delta_step(S_t: torch.Tensor, K: torch.Tensor, sigma: float, tau: float) -> torch.Tensor:
    """A single time-step BS delta (helper used by ``BSStaticDelta``)."""
    if tau <= 1e-10:
        return torch.where(S_t > K, torch.ones_like(S_t), torch.zeros_like(S_t))
    sqrt_tau = math.sqrt(tau)
    d1 = (torch.log(S_t / K) + 0.5 * sigma * sigma * tau) / (sigma * sqrt_tau)
    return _norm_cdf(d1)
