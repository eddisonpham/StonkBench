from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput


class ModelAdapter(ABC):
    """Adapter contract for model-specific training and generation shims.

    Optional checkpoint regeneration contract:
        ``load_state(checkpoints)`` rehydrates the in-memory state from one or
        more ``{channel, model_name, model_state_dict}`` checkpoint files so the
        evaluator can regenerate samples without retraining. Adapters that don't
        support this contract leave ``can_regenerate_from_checkpoint`` at its
        default ``False``; ``load_state`` then raises ``NotImplementedError``.
    """

    model_name: str = "unknown_model"
    can_regenerate_from_checkpoint: bool = False
    supports_arbitrary_generation: bool = False

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

    def load_state(self, checkpoints: List[Path]) -> Dict[str, Any]:
        """Restore in-memory state from per-channel ``.pt`` checkpoints.

        Adapters that serialize all of their internal state in checkpoint files
        override this to set ``can_regenerate_from_checkpoint = True`` on the
        class. The default raises so the evaluator's regenerate flow can catch
        and fall back to evaluating a pre-existing artifact.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support checkpoint regeneration."
        )

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(-1)
        if x.ndim == 3:
            return x
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")

