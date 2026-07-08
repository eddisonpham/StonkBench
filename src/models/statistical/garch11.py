import torch
import numpy as np
from arch import arch_model

from src.models.base.base_model import StatisticalModel


class GARCH11(StatisticalModel):
    def __init__(self):
        super().__init__()
        self.mu = None
        self.omega = None
        self.alpha = None
        self.beta = None
        self.num_channels = 0

    def fit(self, log_returns: torch.Tensor) -> None:
        data = log_returns
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        if data.ndim != 2:
            raise ValueError(f"GARCH11 expects input shaped (L,) or (L, C), got {tuple(log_returns.shape)}")

        self.num_channels = data.shape[1]
        mu_vals = []
        omega_vals = []
        alpha_vals = []
        beta_vals = []

        for c in range(self.num_channels):
            channel_np = data[:, c].detach().cpu().numpy()
            am = arch_model(
                channel_np,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
                rescale=False,
            )
            model_fit = am.fit(disp="off")
            mu_vals.append(float(model_fit.params["mu"]))
            omega_vals.append(float(model_fit.params["omega"]))
            alpha_vals.append(float(model_fit.params["alpha[1]"]))
            beta_vals.append(float(model_fit.params["beta[1]"]))

        self.mu = torch.tensor(mu_vals, dtype=torch.float32)
        self.omega = torch.tensor(omega_vals, dtype=torch.float32)
        self.alpha = torch.tensor(alpha_vals, dtype=torch.float32)
        self.beta = torch.tensor(beta_vals, dtype=torch.float32)
        print(f"GARCH11 fitted with {self.num_channels} channel(s)")

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.mu is None or self.omega is None or self.alpha is None or self.beta is None:
            raise RuntimeError("Call fit() before generate().")

        log_returns = torch.zeros((num_samples, generation_length, self.num_channels), dtype=self.mu.dtype)
        sigma2 = torch.zeros((num_samples, generation_length, self.num_channels), dtype=self.mu.dtype)
        epsilon = torch.zeros((num_samples, generation_length, self.num_channels), dtype=self.mu.dtype)

        denom = torch.clamp(1 - self.alpha - self.beta, min=1e-8)
        sigma2[:, 0, :] = self.omega.unsqueeze(0) / denom.unsqueeze(0)
        epsilon[:, 0, :] = torch.sqrt(torch.clamp(sigma2[:, 0, :], min=1e-12)) * torch.randn(
            num_samples, self.num_channels, dtype=self.mu.dtype
        )
        log_returns[:, 0, :] = self.mu.unsqueeze(0) + epsilon[:, 0, :]

        for t in range(1, generation_length):
            sigma2[:, t, :] = (
                self.omega.unsqueeze(0)
                + self.alpha.unsqueeze(0) * epsilon[:, t - 1, :] ** 2
                + self.beta.unsqueeze(0) * sigma2[:, t - 1, :]
            )
            epsilon[:, t, :] = torch.sqrt(torch.clamp(sigma2[:, t, :], min=1e-12)) * torch.randn(
                num_samples, self.num_channels, dtype=self.mu.dtype
            )
            log_returns[:, t, :] = self.mu.unsqueeze(0) + epsilon[:, t, :]

        if self.num_channels == 1:
            return log_returns.squeeze(-1)
        return log_returns
