"""Base class for all §6 policies.

A policy is a trainable model that maps an input window to a holding/
position trajectory. Every implementation must conform to:

- ``fit(X_train, *, num_epochs, batch_size, learning_rate, ...)`` —
  trains and (optionally) checkpoints the weights.
- ``predict(X)`` — returns a tensor of holdings of shape expected by the
  parent :class:`~src.utility.tasks.base.BaseUtilityTask`.
- ``save(path)`` / ``load(path)`` — pickle the model's parameters.

This base class does not assume any particular network — deep policies
inherit additionally from ``torch.nn.Module``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

import torch


class BaseUtilityPolicy(ABC):
    """Generic train-then-predict contract used by the §6 protocol runner."""

    task_name: str = "base"
    family: str = "base"  # used for tag/group reporting

    @abstractmethod
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
    ) -> "BaseUtilityPolicy":
        """Train the policy. Returns ``self`` for chaining."""

    @abstractmethod
    def predict(self, test_data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return a holdings/positions tensor over time."""

    def parameters_state(self) -> Dict[str, torch.Tensor]:
        """Default implementation: returns ``state_dict`` if nn.Module, else empty."""
        if isinstance(self, torch.nn.Module):
            return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
        return {}

    def save(self, path: str) -> None:
        """Default: dump state dict to disk. Subclasses may override."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.parameters_state(), path)

    def load(self, path: str) -> "BaseUtilityPolicy":
        state = torch.load(path, map_location="cpu")
        if isinstance(self, torch.nn.Module):
            self.load_state_dict(state)
        return self
