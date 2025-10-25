import torch
import numpy as np
from src.models.base.base_model import ParametricModel

class BlockBootstrap(ParametricModel):
    """
    Block Bootstrap model for multichannel time series generation.

    Assumptions:
      - Input array shape: (length, num_channels)
      - Input data is already in log returns (no internal preprocessing)
      - All channels are feature signals (no timestamp channel)
      - Evenly spaced time steps (linear)
      - Resamples contiguous blocks to preserve short-term dependencies

    Args:
        length (int): Length of the original series.
        num_channels (int): Number of channels (features).
        block_size (int, optional): Size of each block for resampling. Defaults to length.
        device (str): 'cpu' or 'cuda'.
    """

    def __init__(self, length: int, num_channels: int, block_size: int = None, device: str = "cpu"):
        super().__init__()
        self.length = length
        self.num_channels = num_channels
        self.block_size = block_size if block_size is not None else length
        self.device = device
        self.data = None

    def fit(self, data: torch.Tensor):
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        self.data = data.to(self.device)
        self.length, self.num_channels = self.data.shape
        return self

    def _resample_once(self, output_length: int = None, rng: np.random.Generator = None) -> torch.Tensor:
        """
        Generate one bootstrap sample by resampling contiguous blocks.
        """
        if output_length is None:
            output_length = self.length

        if rng is None:
            rng = np.random.default_rng()

        samples = []
        total_needed = output_length
        block_size = min(self.block_size, self.length)

        while total_needed > 0:
            start = rng.integers(0, self.length - block_size + 1)
            end = start + min(block_size, total_needed)
            samples.append(self.data[start:end].clone())
            total_needed -= (end - start)

        return torch.cat(samples, dim=0)[:output_length]

    def generate(self, num_samples: int, output_length: int = None, seed: int = None) -> torch.Tensor:
        """
        Generate new log returns time series using block bootstrap.

        Returns:
            torch.Tensor: Generated log returns data of shape
                          (num_samples, output_length, num_channels)
        """
        if self.data is None:
            raise ValueError("Model must be fitted before calling generate().")

        if output_length is None:
            output_length = self.length

        rng = np.random.default_rng(seed)

        generated = [self._resample_once(output_length, rng=rng) for _ in range(num_samples)]
        return torch.stack(generated, dim=0)
