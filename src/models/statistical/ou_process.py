import torch
import numpy as np

from src.models.base.base_model import StatisticalModel


class OUProcess(StatisticalModel):
    def __init__(self):
        super().__init__()
        self.mu = None
        self.theta = None
        self.sigma = None
        self.num_channels = 0

    def fit(self, log_returns: torch.Tensor) -> None:
        data = log_returns
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        if data.ndim != 2:
            raise ValueError(f"OU expects input shaped (L,) or (L, C), got {tuple(log_returns.shape)}")
        if data.shape[0] < 3:
            raise ValueError("OU fitting requires at least 3 observations.")

        self.num_channels = data.shape[1]
        mu_vals = []
        theta_vals = []
        sigma_vals = []

        for c in range(self.num_channels):
            x = data[:, c]
            x_t = x[:-1]
            x_tp1 = x[1:]
            a = torch.stack([x_t, torch.ones_like(x_t)], dim=1)
            sol = torch.linalg.lstsq(a, x_tp1.unsqueeze(1)).solution.squeeze()
            phi = torch.clamp(sol[0], 1e-6, 1 - 1e-6)
            intercept = sol[1]

            theta = -torch.log(phi)
            mu = intercept / torch.clamp(1 - phi, min=1e-8)

            residuals = x_tp1 - (phi * x_t + intercept)
            sigma_e = torch.std(residuals, correction=1)
            denom = torch.clamp(1 - phi**2, min=1e-10)
            sigma = torch.clamp(sigma_e * torch.sqrt(2 * theta / denom), min=1e-8)

            mu_vals.append(mu)
            theta_vals.append(theta)
            sigma_vals.append(sigma)

        self.mu = torch.stack(mu_vals)
        self.theta = torch.stack(theta_vals)
        self.sigma = torch.stack(sigma_vals)
        print(f"OU fitted with {self.num_channels} channel(s)")

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.mu is None or self.theta is None or self.sigma is None:
            raise RuntimeError("Call fit() before generate().")

        log_returns = torch.zeros((num_samples, generation_length, self.num_channels), dtype=self.mu.dtype)
        phi = torch.exp(-self.theta)
        mu_term = self.mu * (1 - phi)
        sigma_term = self.sigma * torch.sqrt((1 - phi**2) / torch.clamp(2 * self.theta, min=1e-10))
        log_returns[:, 0, :] = self.mu.unsqueeze(0) + sigma_term.unsqueeze(0) * torch.randn(
            num_samples, self.num_channels, dtype=self.mu.dtype
        )

        for t in range(1, generation_length):
            noise = torch.randn(num_samples, self.num_channels, dtype=self.mu.dtype)
            log_returns[:, t, :] = (
                log_returns[:, t - 1, :] * phi.unsqueeze(0)
                + mu_term.unsqueeze(0)
                + sigma_term.unsqueeze(0) * noise
            )

        if self.num_channels == 1:
            return log_returns.squeeze(-1)
        return log_returns
