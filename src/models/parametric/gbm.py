import torch
import numpy as np
from typing import Optional

from src.models.base.base_model import ParametricModel


class GeometricBrownianMotion(ParametricModel):
    def __init__(self, length: int, num_channels: int, device='cpu'):
        super().__init__(length, num_channels)
        self.device = device
        self.mu = torch.zeros((self.num_channels,), device=self.device)
        self.sigma = torch.ones((self.num_channels,), device=self.device) * 1e-6
        self.fitted_data = None

    def fit(self, log_returns):
        log_returns = log_returns.to(device=self.device).clone()
        self.mu = torch.mean(log_returns, dim=0)
        self.sigma = torch.std(log_returns, dim=0, unbiased=True).clamp(min=1e-12)

    def generate(self, num_samples: int, seq_length: int, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        N = self.num_channels
        Z = torch.randn((num_samples, seq_length, N), device=self.device)
        mu_term = self.mu.unsqueeze(0).unsqueeze(0)
        sigma_term = self.sigma.unsqueeze(0).unsqueeze(0)
        log_returns = mu_term + sigma_term * Z
        return log_returns
