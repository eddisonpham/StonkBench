"""Base class for downstream tasks (§6.2/§6.3/§6.4).

A task encapsulates:

- how training data is prepared (e.g. 7-moneyness aug, channel slicing);
- which policy it builds, on which channels (per-channel vs whole);
- how its trained policy is scored on a held-out test window — including
  per-period PnL computation (toggleable between log and simple returns);
- which option/extra figures (e.g. σ, premium, hedging loss) it reports
  beyond U1–U5.

The per-task loss is owned by the policy (the network). The task only
governs data flow + reporting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


class BaseUtilityTask(ABC):
    task_name: str = "base"
    policy_family: str = "base"
    is_per_channel: bool = False

    def __init__(self, pnl_returns: str = "simple"):
        if pnl_returns not in ("simple", "log"):
            raise ValueError(f"pnl_returns must be simple or log, got {pnl_returns!r}")
        self.pnl_returns = pnl_returns

    # ------------------------------------------------------------------ #
    # Data prep                                                           #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def prepare_training(
        self,
        log_returns: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        """Return task-specific training payload (often a dict)."""

    # ------------------------------------------------------------------ #
    # Policy                                                              #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build_policy(self, **kwargs: Any) -> Any:
        """Return a fresh, untrained policy instance."""

    # ------------------------------------------------------------------ #
    # Evaluation → per-period PnL                                         #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def predict_period_pnl(
        self,
        policy,
        test_windows: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return per-period PnL for each test window: shape ``(N, L-1)``."""

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #
    def extras(
        self,
        policy,
        test_windows: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Optional task-specific extras (e.g. hedging loss)."""
        return {}
