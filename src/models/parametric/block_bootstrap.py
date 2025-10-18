import torch
import numpy as np


class BlockBootstrap:
    """
    Block Bootstrap model for multichannel time series generation.
    
    Assumptions:
      - Input array shape: (length, num_channels)
      - Evenly spaced time steps
      - Resamples contiguous blocks to preserve short-term dependencies

    Args:
        length (int): Length of the original series.
        num_channels (int): Number of channels (features).
        block_size (int): Size of each block used for resampling.
        device (str): 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        length: int,
        num_channels: int,
        block_size: int = 10,
        device: str = "cpu",
    ):
        self.length = length
        self.num_channels = num_channels
        self.block_size = block_size
        self.device = device
        self.data = None

    def fit(self, data: torch.Tensor):
        """
        Fit the model by storing the original time series.
        """
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        self.data = data.to(self.device)
        self.length, self.num_channels = self.X.shape
        return self

    def _resample_once(self) -> torch.Tensor:
        """
        Generate one bootstrap sample by resampling contiguous blocks.
        """
        samples = []
        total_needed = self.length

        while total_needed > 0:
            start = np.random.randint(0, self.length - self.block_size)
            end = start + min(self.block_size, total_needed)
            samples.append(self.X[start:end])
            total_needed -= (end - start)

        return torch.cat(samples, dim=0)

    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Generate new time series using block bootstrap.

        Args:
            num_samples (int): Number of bootstrap samples to generate.

        Returns:
            torch.Tensor: Generated data of shape (num_samples, length, num_channels)
        """
        if self.X is None:
            raise ValueError("Model must be fitted before calling generate().")

        generated = []
        for _ in range(num_samples):
            sample = self._resample_once()
            generated.append(sample)

        return torch.stack(generated, dim=0)
