import torch
import numpy as np

from src.models.base.base_model import StatisticalModel


class BlockBootstrap(StatisticalModel):
    """
    Moving block bootstrap for univariate or multivariate log-return series.

    Multivariate samples preserve cross-channel dependence within each resampled block.
    """

    def __init__(self, block_size: int = 32):
        super().__init__()
        self.block_size = block_size
        self.log_returns: torch.Tensor | None = None
        self.multivariate = False

    def fit(self, data: torch.Tensor) -> None:
        if data.ndim == 2 and data.shape[1] == 1:
            data = data.squeeze(-1)
        if data.ndim not in (1, 2):
            raise ValueError(f"Expected data shaped (L,) or (L, C), got {tuple(data.shape)}")
        self.log_returns = data
        self.multivariate = data.ndim == 2

    def _sample_indices(self, total_time_steps: int, generation_length: int) -> list[int]:
        if self.log_returns is None:
            raise RuntimeError("Call fit() before generate().")
        if total_time_steps < self.block_size:
            raise ValueError(
                f"Training series length ({total_time_steps}) must be >= block_size ({self.block_size})."
            )
        num_blocks = int(np.ceil(generation_length / self.block_size))
        max_start = total_time_steps - self.block_size + 1
        idxs: list[int] = []
        for _ in range(num_blocks):
            start_idx = torch.randint(0, max_start, (1,)).item()
            idxs.extend(range(start_idx, start_idx + self.block_size))
        return idxs[:generation_length]

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        if self.log_returns is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        np.random.seed(seed)
        total_time_steps = self.log_returns.shape[0]

        if self.multivariate:
            num_channels = self.log_returns.shape[1]
            samples = torch.zeros((num_samples, generation_length, num_channels), dtype=self.log_returns.dtype)
            for sample_idx in range(num_samples):
                idxs = self._sample_indices(total_time_steps, generation_length)
                samples[sample_idx] = self.log_returns[idxs]
            return samples

        samples = torch.zeros((num_samples, generation_length), dtype=self.log_returns.dtype)
        for sample_idx in range(num_samples):
            idxs = self._sample_indices(total_time_steps, generation_length)
            samples[sample_idx] = self.log_returns[idxs]
        return samples
