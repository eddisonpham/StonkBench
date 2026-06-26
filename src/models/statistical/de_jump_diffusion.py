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
        self.num_channels = 0

    def fit(self, log_returns: torch.Tensor) -> None:
        data = log_returns
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        if data.ndim != 2:
            raise ValueError(f"DEJD expects input shaped (L,) or (L, C), got {tuple(log_returns.shape)}")

        self.num_channels = data.shape[1]
        mu_vals = []
        sigma_vals = []
        lam_vals = []
        p_vals = []
        eta1_vals = []
        eta2_vals = []
        kappa_vals = []

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
        print(f"DEJD fitted with {self.num_channels} channel(s)")

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if any(v is None for v in [self.mu, self.sigma, self.lam, self.p, self.eta1, self.eta2, self.kappa]):
            raise RuntimeError("Call fit() before generate().")

        log_returns = torch.zeros((num_samples, generation_length, self.num_channels), dtype=self.mu.dtype)
        for c in range(self.num_channels):
            drift = self.mu[c] - 0.5 * self.sigma[c] ** 2 - self.kappa[c] * self.lam[c]
            diffusion = self.sigma[c] * torch.randn(num_samples, generation_length, dtype=self.mu.dtype)

            num_jumps = torch.poisson(
                torch.full((num_samples, generation_length), float(self.lam[c]), dtype=self.mu.dtype)
            )
            jump_sign = torch.rand(num_samples, generation_length, dtype=self.mu.dtype)
            rand_vals = torch.rand(num_samples, generation_length, dtype=self.mu.dtype)

            pos_jump_sizes = -torch.log(torch.clamp(1 - rand_vals, min=1e-12)) / self.eta1[c]
            neg_jump_sizes = torch.log(torch.clamp(rand_vals, min=1e-12)) / self.eta2[c]
            chosen_jump_sizes = torch.where(jump_sign < self.p[c], pos_jump_sizes, neg_jump_sizes)
            jumps = num_jumps * chosen_jump_sizes

            log_returns[:, :, c] = drift + diffusion + jumps

        if self.num_channels == 1:
            return log_returns.squeeze(-1)
        return log_returns