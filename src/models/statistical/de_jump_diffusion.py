"""Multivariate Double-Exponential Jump-Diffusion (Kou 2002).

Per-channel: independent estimate of (μ, σ, λ, p, η1, η2) by median-threshold jump
detection per channel (identical to the original univariate DEJD model).

Cross-channel: at generate-time, an independent standard-normal tensor (R, L, C) is
correlated via the Cholesky factor of the empirical covariance matrix of training
data, mirroring the Merton multivariate upgrade. Jumps remain per-channel independent
because the Poisson processes have independent increment channels.
"""
from __future__ import annotations

import torch
import numpy as np

from src.models.base.base_model import StatisticalModel


class DoubleExponentialJumpDiffusion(StatisticalModel):
    def __init__(self):
        super().__init__()
        self.mu = None
        self.sigma = None
        self.lam = None
        self.p = None
        self.eta1 = None
        self.eta2 = None
        self.kappa = None
        self.chol_factor = None  # (C, C) Cholesky factor of empirical covariance
        self.num_channels = 0

    def fit(self, log_returns: torch.Tensor) -> None:
        data = log_returns
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        if data.ndim != 2:
            raise ValueError(
                f"DEJD expects input shaped (L,) or (L, C), got {tuple(log_returns.shape)}"
            )

        self.num_channels = data.shape[1]
        mu_vals: list[torch.Tensor] = []
        sigma_vals: list[torch.Tensor] = []
        lam_vals: list[torch.Tensor] = []
        p_vals: list[torch.Tensor] = []
        eta1_vals: list[torch.Tensor] = []
        eta2_vals: list[torch.Tensor] = []
        kappa_vals: list[torch.Tensor] = []

        for c in range(self.num_channels):
            x = data[:, c]
            total = x.shape[0]
            jump_threshold = 3.0

            abs_median = torch.median(torch.abs(x))
            threshold = jump_threshold * torch.clamp(abs_median, min=1e-8)
            small_mask = torch.abs(x) < threshold
            diffusion_returns = x[small_mask]
            sigma = torch.clamp(torch.std(diffusion_returns, unbiased=True), min=1e-8)

            jumps = x[~small_mask]
            lam = float(jumps.shape[0]) / float(max(total, 1))
            pos_jumps = jumps[jumps > 0]
            neg_jumps = jumps[jumps < 0]

            p = float(pos_jumps.shape[0] / max(jumps.shape[0], 1))
            eta1 = float(1.0 / torch.clamp(pos_jumps.mean(), min=1e-8)) if pos_jumps.shape[0] > 0 else 1.5
            eta2 = float(-1.0 / torch.clamp(neg_jumps.mean(), max=-1e-8)) if neg_jumps.shape[0] > 0 else 1.5
            eta1 = max(eta1, 1.0001)
            eta2 = max(eta2, 1e-4)

            kappa = (p * eta1 / (eta1 - 1.0)) + ((1.0 - p) * eta2 / (eta2 + 1.0))
            mu = float(torch.mean(x) + 0.5 * sigma**2 + kappa * lam)

            mu_vals.append(torch.tensor(mu, dtype=x.dtype))
            sigma_vals.append(sigma)
            lam_vals.append(torch.tensor(lam, dtype=x.dtype))
            p_vals.append(torch.tensor(p, dtype=x.dtype))
            eta1_vals.append(torch.tensor(eta1, dtype=x.dtype))
            eta2_vals.append(torch.tensor(eta2, dtype=x.dtype))
            kappa_vals.append(torch.tensor(kappa, dtype=x.dtype))

        self.mu = torch.stack(mu_vals)
        self.sigma = torch.stack(sigma_vals)
        self.lam = torch.stack(lam_vals)
        self.p = torch.stack(p_vals)
        self.eta1 = torch.stack(eta1_vals)
        self.eta2 = torch.stack(eta2_vals)
        self.kappa = torch.stack(kappa_vals)

        # Multivariate coupling: Cholesky factor of empirical covariance.
        cov = (
            torch.cov(data.T)
            if self.num_channels > 1
            else torch.tensor([[torch.var(data[:, 0], unbiased=True)]], dtype=data.dtype)
        )
        cov = cov + 1e-6 * torch.eye(self.num_channels, dtype=cov.dtype)
        try:
            self.chol_factor = torch.linalg.cholesky(cov.to(torch.float64)).to(torch.float32)
        except Exception:
            diag_var = torch.clamp(torch.diag(cov), min=1e-8)
            self.chol_factor = torch.diag(torch.sqrt(diag_var)).to(torch.float32)
        print(
            f"DEJD (multivariate) fitted with {self.num_channels} channel(s); "
            f"diffusion chunks via Cholesky-of-cov"
        )

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.chol_factor is None or any(
            v is None
            for v in [self.mu, self.sigma, self.lam, self.p, self.eta1, self.eta2, self.kappa]
        ):
            raise RuntimeError("Call fit() before generate().")

        # Multivariate correlated diffusion draws.
        z_indep = torch.randn(num_samples, generation_length, self.num_channels, dtype=self.mu.dtype)
        z_corr = torch.einsum("rlk,kc->rlc", z_indep, self.chol_factor)
        diffusion = z_corr * self.sigma.view(1, 1, -1)
        drift = (self.mu - 0.5 * self.sigma**2 - self.lam * self.kappa).view(1, 1, -1)
        log_returns = drift + diffusion

        # Per-channel independent double-exponential jumps.
        for c in range(self.num_channels):
            lam_c = self.lam[c]
            p_c = self.p[c]
            eta1_c = self.eta1[c]
            eta2_c = self.eta2[c]

            num_jumps = torch.poisson(
                torch.full((num_samples, generation_length), float(lam_c), dtype=self.mu.dtype)
            )
            jump_sign = torch.rand(num_samples, generation_length, dtype=self.mu.dtype)
            rand_vals = torch.rand(num_samples, generation_length, dtype=self.mu.dtype)

            pos_jump_sizes = -torch.log(torch.clamp(1 - rand_vals, min=1e-12)) / eta1_c
            neg_jump_sizes = torch.log(torch.clamp(rand_vals, min=1e-12)) / eta2_c
            chosen_jump_sizes = torch.where(jump_sign < p_c, pos_jump_sizes, neg_jump_sizes)
            jumps = num_jumps * chosen_jump_sizes

            log_returns[:, :, c] = log_returns[:, :, c] + jumps

        if self.num_channels == 1:
            return log_returns.squeeze(-1)
        return log_returns
