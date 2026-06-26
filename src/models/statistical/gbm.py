import torch
import numpy as np

from src.models.base.base_model import StatisticalModel


class GeometricBrownianMotion(StatisticalModel):
    def __init__(self):
        super().__init__()
        self.mu = None
        self.sigma = None
        self.num_channels = 0

    def fit(self, log_returns: torch.Tensor) -> None:
        data = log_returns
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        if data.ndim != 2:
            raise ValueError(f"GBM expects input shaped (L,) or (L, C), got {tuple(log_returns.shape)}")

        self.num_channels = data.shape[1]
        self.mu = data.mean(dim=0)
        self.sigma = torch.clamp(data.std(dim=0, unbiased=True), min=1e-8)
        print(f"GBM fitted with {self.num_channels} channel(s)")

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Call fit() before generate().")

        z = torch.randn(num_samples, generation_length, self.num_channels, dtype=self.mu.dtype)
        drift = self.mu - 0.5 * self.sigma**2
        log_returns = drift.view(1, 1, -1) + self.sigma.view(1, 1, -1) * z
        if self.num_channels == 1:
            return log_returns.squeeze(-1)
        return log_returns

