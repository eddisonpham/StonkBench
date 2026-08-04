"""Stationary Block Bootstrap (Politis & Romano 1994) — multivariate variant.

Block lengths are i.i.d. Geometric(p) → non-uniform block boundaries that preserve
stationarity better than the Moving Block Bootstrap (Politis & Romano 1992; uniform
block length). Each resampled block preserves cross-channel dependence — so the
multivariate output has the same cross-channel correlation structure as a contiguous
slice of the training series.
"""
from __future__ import annotations

import torch
import numpy as np

from src.models.base.base_model import StatisticalModel


class StationaryBlockBootstrap(StatisticalModel):
    """Multivariate Stationary Block Bootstrap.

    Parameters
    ----------
    expected_block_length : float
        Average block length in time steps. Internally the geometric parameter
        is p = 1 / expected_block_length, so E[block length] = 1/p.
    """

    def __init__(self, expected_block_length: float = 32.0):
        super().__init__()
        if expected_block_length <= 0:
            raise ValueError(f"expected_block_length must be > 0, got {expected_block_length}")
        self.expected_block_length = float(expected_block_length)
        self.p = 1.0 / max(self.expected_block_length, 1.0)
        self.log_returns: torch.Tensor | None = None
        self.multivariate = False
        self.num_channels = 0

    def fit(self, data: torch.Tensor) -> None:
        if data.ndim == 2 and data.shape[1] == 1:
            data = data.squeeze(-1)
        if data.ndim not in (1, 2):
            raise ValueError(f"Expected data shaped (L,) or (L, C), got {tuple(data.shape)}")
        self.log_returns = data
        self.multivariate = data.ndim == 2
        self.num_channels = data.shape[1] if data.ndim == 2 else 1
        print(
            f"SBB fitted with {self.num_channels} channel(s); "
            f"expected_block_length={self.expected_block_length}"
        )

    def _sample_indices(self, total_time_steps: int, generation_length: int) -> list[int]:
        """Poly & Romano 1994: pick a random start, sample a Geometric(p) block,
        append that block (mod L for circular wrap-around), repeat until generation_length
        is reached; truncate if overshoot."""
        if self.log_returns is None:
            raise RuntimeError("Call fit() before generate().")
        idxs: list[int] = []
        while len(idxs) < generation_length:
            start_idx = int(torch.randint(0, total_time_steps, (1,)).item())
            # Geometric(p) realization. Sample once to get block length.
            # Use a Python-level Geom call rather than torch.distributions for simplicity.
            block_len = int(np.random.geometric(self.p))
            # Circular wrap-around at the series end (Poly & Romano §3.2).
            idxs.extend((start_idx + k) % total_time_steps for k in range(block_len))
        return idxs[:generation_length]

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        if self.log_returns is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        np.random.seed(seed)
        total_time_steps = self.log_returns.shape[0]

        if self.multivariate:
            num_channels = self.log_returns.shape[1]
            samples = torch.zeros(
                (num_samples, generation_length, num_channels),
                dtype=self.log_returns.dtype,
            )
            for sample_idx in range(num_samples):
                idxs = self._sample_indices(total_time_steps, generation_length)
                samples[sample_idx] = self.log_returns[idxs]
            return samples

        samples = torch.zeros((num_samples, generation_length), dtype=self.log_returns.dtype)
        for sample_idx in range(num_samples):
            idxs = self._sample_indices(total_time_steps, generation_length)
            samples[sample_idx] = self.log_returns[idxs]
        return samples
