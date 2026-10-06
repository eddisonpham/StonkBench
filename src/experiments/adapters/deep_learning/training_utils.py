"""Shared training helpers for DL adapters (validation loss + early stopping)."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.core.contracts import AdapterFitInput
from src.utils.device import resolve_device

__all__ = [
    "EarlyStopping",
    "FitTrainingInfo",
    "TrainingParams",
    "average_loss_over_loader",
    "checkpoint_seq_length",
    "make_loader",
    "parse_training_params",
    "reconcile_gluonts",
    "resolve_device",
    "resolve_final_checkpoint",
]

# ``<model_key>_seq<L>_final.pt`` — the canonical one-checkpoint-per-(model,
# seq_length) naming scheme introduced by the simulation_v2 training runs.
_FINAL_CKPT_SEQ_RE = re.compile(r"seq(\d+)_final\.pt$")


def checkpoint_seq_length(path: Path, default: Optional[int] = None) -> int:
    """Parse ``L`` out of a ``*_seq{L}_final.pt`` checkpoint filename."""
    match = _FINAL_CKPT_SEQ_RE.search(Path(path).name)
    if match:
        return int(match.group(1))
    if default is not None:
        return int(default)
    raise ValueError(f"Cannot derive sequence length from checkpoint name: {Path(path).name}")


def resolve_final_checkpoint(
    checkpoints: Sequence[Path], seq_length: Optional[int] = None
) -> Path:
    """Pick the canonical checkpoint for ``load_state`` out of a manifest.

    ``fit`` writes both a legacy name (``vrnn_checkpoint.pt``) and the
    consolidated ``<model_key>_seq{L}_final.pt``; manifests and evaluator
    regenerate flows may hand us either (or both). Resolution order:

    1. a file ending in ``seq{seq_length}_final.pt`` when ``seq_length`` is given
    2. the single ``*_final.pt`` file, if exactly one exists
    3. the single path in the list
    """
    paths = [Path(p) for p in checkpoints]
    if not paths:
        raise ValueError("load_state received an empty checkpoint list.")
    if seq_length is not None:
        suffix = f"seq{int(seq_length)}_final.pt"
        exact = [p for p in paths if p.name.endswith(suffix)]
        if exact:
            return exact[0]
    finals = [p for p in paths if "_final" in p.name]
    if len(finals) == 1:
        return finals[0]
    if len(paths) == 1:
        return paths[0]
    names = [p.name for p in paths]
    raise ValueError(
        f"Cannot resolve a unique final checkpoint (seq_length={seq_length}) "
        f"from {names}."
    )


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

    def __init__(
        self,
        patience: int = 12,
        min_delta: float = 0.0,
        mode: str = "min",
        min_epochs: int = 0,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.min_epochs = int(min_epochs)
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
        # Do not early-stop before min_epochs (avoids premature GAN/VAE collapse).
        if int(epoch) < self.min_epochs:
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


def reconcile_gluonts(vendor_root: Path, prefer_vendor: bool) -> None:
    """Make ``gluonts`` resolve to the copy this adapter needs.

    Several vendors ship their own ``gluonts`` (e.g. the TimeGrad tree roots a
    vendored copy at ``src/models/deep_learning/timegrad/gluonts``). In a
    single-process run over multiple models the first adapter to import wins
    and later adapters fail with confusing ImportErrors (e.g. ``cannot import
    name 'DataEntry'``). Called before each vendor import:

    * ``prefer_vendor=True``  → drop cached non-vendor ``gluonts*`` modules so
      the vendored copy is imported fresh.
    * ``prefer_vendor=False`` → drop the vendor root from ``sys.path`` and
      purge cached modules from it, so the installed gluonts is used.

    Behavior-neutral when no conflicting modules are cached.
    """
    root = str(Path(vendor_root).resolve())
    if not prefer_vendor:
        sys.path[:] = [
            p for p in sys.path if str(Path(p).resolve()) != root
        ]
    for name in [
        m for m in list(sys.modules) if m == "gluonts" or m.startswith("gluonts.")
    ]:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        f = getattr(mod, "__file__", None) or ""
        if f:
            try:
                under = str(Path(f).resolve()).startswith(root)
            except OSError:
                under = False
        else:
            paths = list(getattr(mod, "__path__", None) or [])
            under = any(str(Path(p).resolve()).startswith(root) for p in paths)
        if under != prefer_vendor:
            del sys.modules[name]


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
