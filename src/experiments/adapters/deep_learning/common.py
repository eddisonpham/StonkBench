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

    Implements ``can_regenerate_from_checkpoint = True`` so the evaluator's
    ``--no_skip_regenerate`` flow can rebuild the training windows in-memory
    from the manifest of one ``{channel, model_name}`` checkpoint per channel
    (the same manifest produced during ``fit``).
    """

    checkpoint_prefix = "channel_model"
    can_regenerate_from_checkpoint = True

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

    def load_state(self, checkpoints: List[Path]) -> Dict[str, Any]:
        """Restore ``base_sequences`` from disk for in-memory regeneration.

        ChannelBootstrapAdapter is the smoke / fallback adapter: ``fit`` saves
        per-channel stubs but the generative state is just the stored
        ``train_windows`` tensor. To regenerate, we look up the canonical
        ``dl_set.pt`` and rebuild ``(N, L, C)`` train windows.
        """
        if not checkpoints:
            raise ValueError("Cannot load_state without checkpoint paths.")
        from src.utils.preprocessed_data_utils import build_batch_from_dl_set, load_dl_set, resolve_dl_set_path

        dl_set = load_dl_set(resolve_dl_set_path())
        batch = build_batch_from_dl_set(dl_set, generation_length=int(dl_set["window_size"]))
        if batch.train_windows is None:
            raise ValueError("dl_set.pt has no train_windows; cannot rebuild for regeneration.")
        self.base_sequences = batch.train_windows.float()
        self.checkpoints = list(checkpoints)
        self._is_fitted = True
        return {"num_channels": int(self.base_sequences.shape[-1]), "restored": True}

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

