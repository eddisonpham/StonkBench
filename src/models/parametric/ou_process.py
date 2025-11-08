import torch
import numpy as np

from src.models.base.base_model import ParametricModel


class OUProcess(ParametricModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.mu = None
        self.theta = None
        self.sigma = None

    def fit(self, log_returns: torch.Tensor) -> None:
        X = log_returns.flatten()
        X_t = X[:-1]
        X_tp1 = X[1:]
        A = torch.stack([X_t, torch.ones_like(X_t)], dim=1)
        sol = torch.linalg.lstsq(A, X_tp1.unsqueeze(1)).solution
        phi, c = sol.squeeze()
        phi = torch.clamp(phi, 1e-6, 1 - 1e-6)
        self.theta = -torch.log(phi)
        self.mu = c / (1 - phi)
        residuals = X_tp1 - (phi * X_t + c)
        sigma_e = torch.std(residuals, correction=1)
        denom = torch.clamp(1 - phi ** 2, min=1e-10)
        self.sigma = sigma_e * torch.sqrt(2 * self.theta / denom)
        print(f"mu: {self.mu.item()}, theta: {self.theta.item()}, sigma: {self.sigma.item()}")

    def generate(self, num_samples: int, generation_length: int):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        log_returns = torch.zeros((num_samples, generation_length))
        phi = torch.exp(-self.theta)
        mu_term = self.mu * (1 - phi)
        sigma_term = self.sigma * torch.sqrt((1 - phi ** 2) / (2 * self.theta))
        log_returns[:, 0] = self.mu + sigma_term * torch.randn(num_samples)
        for t in range(1, generation_length):
            noise = torch.randn(num_samples)
            log_returns[:, t] = (
                log_returns[:, t - 1] * phi
                + mu_term
                + sigma_term * noise
            )

        return log_returns
