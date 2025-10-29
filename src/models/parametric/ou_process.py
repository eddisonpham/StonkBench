import torch
import numpy as np
from src.models.base.base_model import ParametricModel

class OUProcess(ParametricModel):
    def __init__(self, length: int, num_channels: int, delta_t=1.0, device='cpu'):
        super().__init__(length, num_channels)
        self.delta_t = delta_t
        self.device = device
        self.mu = None
        self.theta = None
        self.sigma = None

    def fit(self, log_prices):
        log_prices = log_prices.to(self.device, dtype=torch.float32)
        l, N = log_prices.shape
        mu_list, theta_list, sigma_list = [], [], []

        for i in range(N):
            X = log_prices[:, i]
            X_t, X_tp1 = X[:-1], X[1:]
            A = torch.stack([X_t, torch.ones_like(X_t)], dim=1)
            sol = torch.linalg.lstsq(A, X_tp1.unsqueeze(1)).solution
            phi, c = sol[:2, 0]
            phi = torch.clamp(phi, 1e-8, 0.999999)
            theta = -torch.log(phi) / self.delta_t
            mu = c / (1 - phi)
            residuals = X_tp1 - (phi * X_t + c)
            sigma = torch.std(residuals, correction=1) * torch.sqrt(2 * theta / (1 - phi ** 2))
            mu_list.append(mu)
            theta_list.append(theta)
            sigma_list.append(sigma)

        self.mu = torch.stack(mu_list).to(self.device)
        self.theta = torch.stack(theta_list).to(self.device)
        self.sigma = torch.stack(sigma_list).to(self.device)

    def generate(self, num_samples, seq_length, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        N = self.mu.shape[0]
        l = seq_length if seq_length is not None else self.length
        log_prices = torch.zeros((num_samples, l, N), device=self.device)
        phi = torch.exp(-self.theta * self.delta_t)
        mu_term = self.mu * (1 - phi)
        sigma_term = self.sigma * torch.sqrt((1 - phi ** 2) / (2 * self.theta))
        log_prices[:, 0, :] = self.mu + sigma_term * torch.randn(num_samples, N, device=self.device)
        for t in range(1, l):
            noise = torch.randn(num_samples, N, device=self.device)
            log_prices[:, t, :] = (
                log_prices[:, t - 1, :] * phi
                + mu_term
                + sigma_term * noise
            )

        return log_prices
