from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class StandardBatch:
    """Standardized split tensors for multivariate log-return modeling."""

    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor
    train_initial: torch.Tensor
    valid_initial: torch.Tensor
    test_initial: torch.Tensor
    asset_columns: List[str]
    price_columns: List[str]
    train_windows: Optional[torch.Tensor] = None
    valid_windows: Optional[torch.Tensor] = None
    test_windows: Optional[torch.Tensor] = None
    train_window_initials: Optional[torch.Tensor] = None
    valid_window_initials: Optional[torch.Tensor] = None
    test_window_initials: Optional[torch.Tensor] = None
    inferred_length: Optional[int] = None


@dataclass
class AdapterFitInput:
    """Input object passed to adapters during fit."""

    batch: StandardBatch
    sequence_length: int
    num_epochs: int
    device: str
    seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterGenerateOutput:
    """Standardized generation output for downstream routing/evaluation."""

    data: torch.Tensor  # shape: (R, L, C)
    checkpoints: List[Path] = field(default_factory=list)
    logs: Dict[str, Any] = field(default_factory=dict)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentPaths:
    root: Path
    model_root: Path
    artifacts: Path
    checkpoints: Path
    logs: Path
    metrics: Path

