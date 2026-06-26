import torch
import numpy as np

from src.models.base.base_model import StatisticalModel


class MertonJumpDiffusion(StatisticalModel):
    def __init__(self):
        super().__init__()
        self.mu = None
        self.sigma = None
        self.lam = None
        self.mu_j = None
        self.sigma_j = None
        self.kappa = None
        self.num_channels = 0

    def fit(self, log_returns: torch.Tensor) -> None:
        data = log_returns
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        if data.ndim != 2:
            raise ValueError(f"Merton expects input shaped (L,) or (L, C), got {tuple(log_returns.shape)}")

        self.num_channels = data.shape[1]
        mu_vals = []
        sigma_vals = []
        lam_vals = []
        mu_j_vals = []
        sigma_j_vals = []
        kappa_vals = []

        for c in range(self.num_channels):
            x = data[:, c]
            sigma = torch.clamp(torch.std(x, unbiased=True), min=1e-8)
            threshold = 3.0 * sigma
            jump_mask = torch.abs(x) > threshold
            jumps = x[jump_mask]

            lam = float(jumps.numel()) / float(max(x.numel(), 1))
            if jumps.numel() > 0:
                mu_j = torch.mean(jumps)
                sigma_j = torch.clamp(torch.std(jumps, unbiased=False), min=1e-8)
            else:
                mu_j = torch.tensor(0.0, dtype=x.dtype)
                sigma_j = torch.tensor(0.0, dtype=x.dtype)

            kappa = torch.exp(mu_j + 0.5 * sigma_j**2) - 1.0
            mu = torch.mean(x) + 0.5 * sigma**2 + kappa * lam

            mu_vals.append(mu)
            sigma_vals.append(sigma)
            lam_vals.append(torch.tensor(lam, dtype=x.dtype))
            mu_j_vals.append(mu_j)
            sigma_j_vals.append(sigma_j)
            kappa_vals.append(kappa)

        self.mu = torch.stack(mu_vals)
        self.sigma = torch.stack(sigma_vals)
        self.lam = torch.stack(lam_vals)
        self.mu_j = torch.stack(mu_j_vals)
        self.sigma_j = torch.stack(sigma_j_vals)
        self.kappa = torch.stack(kappa_vals)
        print(f"Merton fitted with {self.num_channels} channel(s)")

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if any(v is None for v in [self.mu, self.sigma, self.lam, self.mu_j, self.sigma_j, self.kappa]):
            raise RuntimeError("Call fit() before generate().")

        log_returns = torch.zeros((num_samples, generation_length, self.num_channels), dtype=self.mu.dtype)
        for c in range(self.num_channels):
            eps = torch.randn(num_samples, generation_length, dtype=self.mu.dtype)
            diffusion = (
                self.mu[c] - 0.5 * self.sigma[c] ** 2 - self.lam[c] * self.kappa[c]
            ) + self.sigma[c] * eps

            num_jumps = torch.poisson(
                torch.full((num_samples, generation_length), float(self.lam[c]), dtype=self.mu.dtype)
            )
            jumps = torch.zeros((num_samples, generation_length), dtype=self.mu.dtype)

            nz = torch.nonzero(num_jumps > 0, as_tuple=False)
            for idx in nz:
                i, t = int(idx[0]), int(idx[1])
                n = int(num_jumps[i, t].item())
                jumps[i, t] = torch.sum(self.mu_j[c] + self.sigma_j[c] * torch.randn(n, dtype=self.mu.dtype))

            log_returns[:, :, c] = diffusion + jumps

        if self.num_channels == 1:
            return log_returns.squeeze(-1)
        return log_returns