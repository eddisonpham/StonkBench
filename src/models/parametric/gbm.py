import torch
import numpy as np

from src.models.base.base_model import ParametricModel


class GeometricBrownianMotion(ParametricModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.mu = None
        self.sigma = None

    def fit(self, log_returns: torch.Tensor) -> None:
        self.mu = torch.mean(log_returns)
        self.sigma = torch.std(log_returns, unbiased=True)
        print(f"mu: {self.mu}, sigma: {self.sigma}")

    def generate(self, num_samples: int, generation_length: int) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        Z = torch.randn(num_samples, generation_length)
        log_returns = (self.mu - 0.5 * self.sigma**2) + self.sigma * Z
        return log_returns

