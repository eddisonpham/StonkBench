"""Paper §6.2 LSTM hedge (moneyness-based).

Autoregressive: at each step ``t``, the LSTM is fed
``(Δ_{t-1}, g̃(t), τ(t))`` and emits ``Δ_t``. The previous holding
``Δ_{t-1}`` becomes an input feature at the next step.

Loss is the *quadratic replication error in moneyness units* (paper eq.):

::

    g̃(t)   = S_t / K
    g̃(L)   = max(S_L / K - 1, 0)
    g'(L)   = g̃(L) - c_0 / K
    R̃      = g'(L) - Σ_t Δ_t · (g̃(t+1) - g̃(t))
    L_Δ     = mean(R̃ ** 2)

The option premium ``c_0`` is computed analytically via Black–Scholes–Merton
(``r=0``, ``σ`` from the *training* inputs). A learnable ``nn.Parameter``
premium is an opt-in flag for diagnostic use.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.utility.policies.base import BaseUtilityPolicy
from src.utils.device import get_device
from src.utility.metrics import safe_exp_cumsum


# --------------------------------------------------------------------------- #
# Black–Scholes–Merton helpers (r=0)                                          #
# --------------------------------------------------------------------------- #
def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(
    S0: torch.Tensor, K: torch.Tensor, sigma: float, T: float, r: float = 0.0
) -> torch.Tensor:
    """Vectorised Black–Scholes–Merton call price (r=0 default per paper)."""
    S0 = S0.float().reshape(-1)
    K = K.float().reshape(-1)
    if sigma <= 0 or T <= 0:
        return torch.clamp(S0 - K, min=0.0)
    sqrtT = math.sqrt(T)
    d1 = (torch.log(S0 / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_delta(
    S_t: torch.Tensor, K: torch.Tensor, sigma: float, tau: torch.Tensor, r: float = 0.0
) -> torch.Tensor:
    """BS delta of a European call: ∂C/∂S = N(d1). ``tau`` is time-to-maturity."""
    S_t = S_t.float()
    K = K.float()
    tau = tau.float().clamp(min=1e-8)
    sqrt_tau = torch.sqrt(tau)
    d1 = (torch.log(S_t / K) + 0.5 * sigma * sigma * tau) / (sigma * sqrt_tau)
    return _norm_cdf(d1)


# --------------------------------------------------------------------------- #
# MoneynessLSTM policy                                                        #
# --------------------------------------------------------------------------- #
class MoneynessLSTM(nn.Module, BaseUtilityPolicy):
    task_name = "options"
    family = "lstm"

    INPUT_DIM = 3  # [Δ_prev, g̃_t, τ_t]

    def __init__(
        self,
        seq_length: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        premium_mode: str = "bs",
        learnable_premium: bool = False,
    ):
        nn.Module.__init__(self)
        BaseUtilityPolicy.__init__(self)
        if premium_mode not in ("bs", "learnable", "zero"):
            raise ValueError(f"premium_mode must be bs|learnable|zero, got {premium_mode!r}")
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.premium_mode = premium_mode
        self.learnable_premium = learnable_premium
        self.device = get_device()

        self.lstm = nn.LSTM(
            input_size=self.INPUT_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        if learnable_premium:
            self.premium_p = nn.Parameter(torch.zeros(1))
        self.to(self.device)

    # ------------------------------------------------------------------ #
    # Forward / predict                                                    #
    # ------------------------------------------------------------------ #
    def predict(
        self,
        test_paths: torch.Tensor,
        K: torch.Tensor,
        c0: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        """Run the autoregressive LSTM to produce Δ of shape ``(N, L)``.

        Args:
            test_paths: prices ``(N, L)`` (or log returns ``(N, L-1)`` —
                in which case prices are reconstructed using ``S_0``).
            K: scalar or ``(N,)`` strike tensor. Moneyness is computed
                against this.
            c0: optional scalar/tensor premium (only used to make
                ``premium`` available externally; not auto-fed to the
                network).
        """
        self.eval()
        paths = test_paths.to(self.device).float()
        N, L = paths.shape
        if K.dim() == 0:
            K = K.expand(N).contiguous()
        elif K.shape[0] == 1:
            K = K.expand(N).contiguous()
        elif K.shape[0] != N:
            raise ValueError(
                f"MoneynessLSTM.predict: K has length {K.shape[0]} but "
                f"there are {N} paths — pass a scalar, shape ({N},), or "
                f"None to use the constructor default."
            )
        K = K.to(self.device).float()

        # Reconstruct prices if log returns were passed.
        if L == self.seq_length - 1:
            # Log returns: derive prices using ``S_0 = K`` (per-window
            # anchor money-back to the strike, matching moneyness = 1).
            S0 = K  # anchor at strike → moneyness at t=0 is exactly 1
            prices = torch.empty(N, L + 1, device=self.device, dtype=torch.float32)
            prices[:, 0] = S0
            prices[:, 1:] = S0.unsqueeze(-1) * safe_exp_cumsum(paths, dim=1)
        else:
            prices = paths

        moneyness = prices / K.unsqueeze(-1)  # (N, L)
        # Paper §6.2: τ_t = 1 - t/L exactly, where L is the option
        # maturity in periods (= ``self.seq_length``). At t=L we want
        # τ=0, so we divide by ``seq_length`` rather than the prices
        # array length (which is L+1 once we prepend ``S_0``).
        denom = max(int(self.seq_length), 1)
        t_idx = torch.arange(prices.shape[1], device=self.device,
                             dtype=torch.float32)
        tau = (1.0 - t_idx / denom).clamp(min=0.0)
        tau = tau.unsqueeze(0).expand(N, -1)

        h, c = None, None
        delta_prev = torch.zeros(N, device=self.device, dtype=torch.float32)
        deltas = []
        for t in range(prices.shape[1]):
            x = torch.stack([delta_prev, moneyness[:, t], tau[:, t]], dim=-1).unsqueeze(1)
            out, (h, c) = self.lstm(x) if h is None else self.lstm(x, (h, c))
            d_t = self.head(out).squeeze(1).squeeze(-1)
            deltas.append(d_t)
            delta_prev = d_t
        return torch.stack(deltas, dim=1)  # (N, L)

    # ------------------------------------------------------------------ #
    # Loss in moneyness units                                             #
    # ------------------------------------------------------------------ #
    def loss(
        self,
        prices_batch: torch.Tensor,
        K_batch: torch.Tensor,
        c0_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Quadratic replication error in moneyness units (paper eq.)."""
        deltas = self.predict(prices_batch, K_batch, c0=c0_batch)  # (N, L)
        g_t = prices_batch / K_batch.unsqueeze(-1)  # (N, L)
        g_diff = g_t[:, 1:] - g_t[:, :-1]  # (N, L-1)
        # The first L-1 deltas act on L-1 price moves; final delta unused.
        if self.premium_mode == "bs":
            c0_used = c0_batch
        elif self.premium_mode == "zero":
            c0_used = torch.zeros_like(c0_batch)
        else:
            c0_used = self.premium_p.expand_as(c0_batch)

        # Replication residual in moneyness units (using Δ_0..Δ_{L-2})
        # If premium is per-K: c0/K is the per-unit normalized premium.
        g_terminal = torch.clamp(prices_batch[:, -1] / K_batch - 1.0, min=0.0)
        g_diff_normalized = c0_used / K_batch  # (N,) per path
        residual = g_terminal - g_diff_normalized - (deltas[:, :-1] * g_diff).sum(dim=1)
        return (residual ** 2).mean()

    # ------------------------------------------------------------------ #
    # Training                                                            #
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
    ) -> "MoneynessLSTM":
        """``train_data`` is a dict from :meth:`OptionsTask.augment_training`.

        Required keys: ``prices`` ``(N, L)``, ``K`` ``(N,)``,
        ``c0`` ``(N,)``.
        """
        nn.Module.train(self)
        self.train()
        if not isinstance(train_data, dict):
            raise TypeError(
                "MoneynessLSTM.fit requires the dict payload from "
                "OptionsTask.augment_training (keys: prices, K, c0)."
            )
        prices = train_data["prices"].to(self.device).float()
        K = train_data["K"].to(self.device).float()
        c0 = train_data["c0"].to(self.device).float()

        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        n = prices.shape[0]
        last_loss = float("nan")
        for epoch in range(num_epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            n_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = self.loss(prices[idx], K[idx], c0[idx])
                loss.backward()
                opt.step()
                total += loss.item()
                n_batches += 1
            last_loss = total / max(1, n_batches)
            if verbose and (epoch % max(1, num_epochs // 5) == 0 or epoch == num_epochs - 1):
                print(f"[MoneynessLSTM] epoch {epoch + 1}/{num_epochs}  loss={last_loss:.6f}")
        return self
