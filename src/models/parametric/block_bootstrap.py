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
        super().__init__()
        self.length = length
        self.num_channels = num_channels
        self.block_size = block_size
        self.device = device
        self.data = None

    def fit(self, data: torch.Tensor):
        """
        Fit the model by storing the original log returns time series.
        
        Args:
            data: Input log returns time series of shape (length, num_channels)
        """
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        self.data = data.to(self.device)
        self.length, self.num_channels = self.data.shape
        return self

    def _resample_once(self, output_length: int = None) -> torch.Tensor:
        """
        Generate one bootstrap sample by resampling contiguous blocks.
        
        Args:
            output_length (int, optional): Desired output length.
                Defaults to the original fitted series length.

        Returns:
            torch.Tensor: Resampled log returns time series of shape
                          (output_length, num_channels)
        """
        if output_length is None:
            output_length = self.length

        samples = []
        total_needed = output_length

        while total_needed > 0:
            start = np.random.randint(0, self.length - self.block_size + 1)
            end = start + min(self.block_size, total_needed)
            samples.append(self.data[start:end])
            total_needed -= (end - start)

        result = torch.cat(samples, dim=0)[:output_length]
        return result

    def generate(self, num_samples: int, output_length: int = None) -> torch.Tensor:
        """
        Generate new log returns time series using block bootstrap.

        Args:
            num_samples (int): Number of bootstrap samples to generate.
            output_length (int, optional): Desired length of each sample.
                Defaults to the fitted series length.

        Returns:
            torch.Tensor: Generated log returns data of shape
                          (num_samples, output_length, num_channels)
        """
        if self.data is None:
            raise ValueError("Model must be fitted before calling generate().")

        if output_length is None:
            output_length = self.length

        generated = []
        for _ in range(num_samples):
            sample = self._resample_once(output_length)
            generated.append(sample)

        return torch.stack(generated, dim=0)
