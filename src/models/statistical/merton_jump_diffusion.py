"""Multivariate Merton Jump-Diffusion.

Per-channel: independent estimate of (μ, σ, λ, μ_j, σ_j) from per-channel absolute-jump
detection logic identical to the original univariate Merton model.

Cross-channel: at generate-time, an independent standard-normal tensor (R, L, C) is
multiplied by the Cholesky factor of the empirical covariance matrix of the training
data → correlated diffusion shocks across channels. This preserves cross-channel
correlation in the diffusion component while keeping Poisson jumps independent per
channel (independent increment assumption).

Output: (R, L, C) for multivariate, (R, L) for univariate.
"""
from __future__ import annotations

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
        self.chol_factor = None  # (C, C) Cholesky factor of empirical covariance
        self.num_channels = 0

    def fit(self, log_returns: torch.Tensor) -> None:
        data = log_returns
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        if data.ndim != 2:
            raise ValueError(f"Merton expects input shaped (L,) or (L, C), got {tuple(log_returns.shape)}")

        self.num_channels = data.shape[1]
        mu_vals: list[torch.Tensor] = []
        sigma_vals: list[torch.Tensor] = []
        lam_vals: list[torch.Tensor] = []
        mu_j_vals: list[torch.Tensor] = []
        sigma_j_vals: list[torch.Tensor] = []
        kappa_vals: list[torch.Tensor] = []

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

        # Multivariate coupling: Cholesky factor of empirical covariance.
        cov = torch.cov(data.T) if self.num_channels > 1 else torch.tensor(
            [[torch.var(data[:, 0], unbiased=True)]], dtype=data.dtype
        )
        # Tiny ridge to enforce PSD on near-degenerate series.
        cov = cov + 1e-6 * torch.eye(self.num_channels, dtype=cov.dtype)
        try:
            self.chol_factor = torch.linalg.cholesky(cov.to(torch.float64)).to(torch.float32)
        except Exception:
            # Fallback: diagonal-only Cholesky (independent channels — preserves per-channel variance).
            diag_var = torch.clamp(torch.diag(cov), min=1e-8)
            self.chol_factor = torch.diag(torch.sqrt(diag_var)).to(torch.float32)
        print(
            f"Merton (multivariate) fitted with {self.num_channels} channel(s); "
            f"diffusion chunks via Cholesky-of-cov"
        )

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.chol_factor is None or any(
            v is None for v in [self.mu, self.sigma, self.lam, self.mu_j, self.sigma_j, self.kappa]
        ):
            raise RuntimeError("Call fit() before generate().")

        # Multivariate diffusion: independent draws correlated via Cholesky.
        # z_indep ~ N(0, 1) → z_corr = z_indep @ L^T  (R, L, C)
        z_indep = torch.randn(num_samples, generation_length, self.num_channels, dtype=self.mu.dtype)
        z_corr = torch.einsum("rlk,kc->rlc", z_indep, self.chol_factor)
        diffusion = z_corr * self.sigma.view(1, 1, -1)
        drift = (self.mu - 0.5 * self.sigma**2 - self.lam * self.kappa).view(1, 1, -1)
        log_returns = drift + diffusion

        # Per-channel independent Poisson jumps.
        for c in range(self.num_channels):
            num_jumps = torch.poisson(
                torch.full((num_samples, generation_length), float(self.lam[c]), dtype=self.mu.dtype)
            )
            jumps = torch.zeros((num_samples, generation_length), dtype=self.mu.dtype)
            nz = torch.nonzero(num_jumps > 0, as_tuple=False)
            for idx in nz:
                i_, t_ = int(idx[0]), int(idx[1])
                n = int(num_jumps[i_, t_].item())
                jumps[i_, t_] = torch.sum(
                    self.mu_j[c] + self.sigma_j[c] * torch.randn(n, dtype=self.mu.dtype)
                )
            log_returns[:, :, c] = log_returns[:, :, c] + jumps

        if self.num_channels == 1:
            return log_returns.squeeze(-1)
        return log_returns
