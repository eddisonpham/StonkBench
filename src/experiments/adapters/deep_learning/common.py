from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.utils.artifact_utils import stitch_sequences


class ChannelBootstrapAdapter(ModelAdapter):
    """
    Lightweight per-channel adapter fallback used when external model APIs are
    not directly pluggable in the benchmark process.

    It preserves standardized shapes and checkpoint routing so orchestration,
    evaluation, and smoke tests remain deterministic.
    """

    checkpoint_prefix = "channel_model"

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.base_sequences: torch.Tensor | None = None
        self.checkpoints: List[Path] = []

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, int]:
        windows = fit_input.batch.train_windows
        if windows is None:
            raise ValueError(f"{self.model_name} requires train_windows in StandardBatch.")
        if windows.ndim != 3:
            raise ValueError("Expected train windows with shape (N, L, C).")
        self.base_sequences = windows.float()
        self.checkpoints = []

        # Save one checkpoint per channel for per-asset univariate adapters.
        num_channels = windows.shape[-1]
        for c in range(num_channels):
            ckpt = checkpoints_dir / f"{self.checkpoint_prefix}_{c + 1}.pt"
            torch.save({"channel": c, "model_name": self.model_name}, ckpt)
            self.checkpoints.append(ckpt)

        self._is_fitted = True
        return {"num_channels": num_channels}

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.base_sequences is None:
            raise RuntimeError("Call fit() before generate().")

        generator = torch.Generator(device=self.base_sequences.device)
        generator.manual_seed(seed)

        n_windows, base_len, n_channels = self.base_sequences.shape
        indices = torch.randint(0, n_windows, (num_samples,), generator=generator)
        sampled = self.base_sequences[indices]  # (R, base_len, C)

        if generation_length != base_len:
            stitched_channels = []
            for c in range(n_channels):
                stitched_c = stitch_sequences(sampled[:, :, c], generation_length, seed + c)
                stitched_channels.append(stitched_c.unsqueeze(-1))
            sampled = torch.cat(stitched_channels, dim=-1)

        return AdapterGenerateOutput(
            data=sampled.float(),
            checkpoints=self.checkpoints,
            logs={"sampling": "window_bootstrap"},
            extra_metadata={"num_channels": n_channels},
        )

