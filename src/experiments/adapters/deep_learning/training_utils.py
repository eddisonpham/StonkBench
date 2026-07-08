"""Shared training helpers for DL adapters (validation loss + early stopping)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.core.contracts import AdapterFitInput
from src.utils.device import resolve_device

__all__ = [
    "EarlyStopping",
    "FitTrainingInfo",
    "TrainingParams",
    "average_loss_over_loader",
    "make_loader",
    "parse_training_params",
    "resolve_device",
    "use_calibration",
]


@dataclass
class TrainingParams:
    max_epochs: int
    patience: int
    learning_rate: float
    batch_size: int


@dataclass
class FitTrainingInfo:
    best_val_loss: float
    best_epoch: int
    stopped_early: bool
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)

    def as_dict(self, **extra: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": int(self.best_epoch),
            "stopped_early": bool(self.stopped_early),
            "train_loss_history": [float(x) for x in self.train_loss_history],
            "val_loss_history": [float(x) for x in self.val_loss_history],
        }
        payload.update(extra)
        return payload


class EarlyStopping:
    """Stop when validation loss fails to improve for `patience` epochs."""

    def __init__(self, patience: int = 12, min_delta: float = 0.0, mode: str = "min") -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best: Optional[float] = None
        self.best_epoch = 0
        self.counter = 0
        self.stopped_early = False

    def step(self, val_loss: float, epoch: int) -> bool:
        improved = self._is_improvement(val_loss)
        if improved:
            self.best = float(val_loss)
            self.best_epoch = int(epoch)
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.stopped_early = True
            return True
        return False

    def _is_improvement(self, val_loss: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return val_loss < self.best - self.min_delta
        return val_loss > self.best + self.min_delta


def parse_training_params(fit_input: AdapterFitInput) -> TrainingParams:
    meta = fit_input.metadata
    # The benchmark entrypoints already choose the intended training budget.
    # Only override it when an explicit HP-search value is provided in metadata.
    default_epochs = max(1, int(fit_input.num_epochs))
    return TrainingParams(
        max_epochs=int(meta.get("max_epochs", default_epochs)),
        patience=int(meta.get("patience", 12)),
        learning_rate=float(meta.get("learning_rate", 1e-3)),
        batch_size=int(meta.get("batch_size", 64)),
    )


def use_calibration(fit_input: AdapterFitInput) -> bool:
    return bool(fit_input.metadata.get("use_calibration", False))


def make_loader(
    windows: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(windows.float())
    effective_batch = max(1, min(batch_size, windows.shape[0]))
    return DataLoader(dataset, batch_size=effective_batch, shuffle=shuffle)


def average_loss_over_loader(compute_batch_loss) -> float:
    total = 0.0
    count = 0
    for loss in compute_batch_loss():
        total += float(loss)
        count += 1
    if count == 0:
        raise ValueError("Cannot compute validation loss on an empty loader.")
    return total / count
