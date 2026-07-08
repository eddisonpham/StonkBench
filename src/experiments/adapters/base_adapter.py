from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import torch

from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput


class ModelAdapter(ABC):
    """Adapter contract for model-specific training and generation shims."""

    model_name: str = "unknown_model"

    def __init__(self) -> None:
        self._is_fitted = False

    @abstractmethod
    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        num_samples: int,
        generation_length: int,
        seed: int,
    ) -> AdapterGenerateOutput:
        raise NotImplementedError

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(-1)
        if x.ndim == 3:
            return x
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")

