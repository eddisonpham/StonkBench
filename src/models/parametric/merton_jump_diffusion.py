import torch
import numpy as np

from src.models.base.base_model import ParametricModel


class MertonJumpDiffusion(ParametricModel):
    def __init__(self, length: int, num_channels: int, delta_t=1.0):
        super().__init__(length, num_channels)
        self.delta_t = delta_t
        self.mu = None
        self.sigma = None
        self.lam = None
        self.mu_j = None
        self.sigma_j = None

    def fit(self, log_returns, jump_threshold=3.0):
        log_returns = np.asarray(log_returns)
        l, N = log_returns.shape
        total_time = l * self.delta_t

        self.mu = np.zeros(N)
        self.sigma = np.zeros(N)
        self.lam = np.zeros(N)
        self.mu_j = np.zeros(N)
        self.sigma_j = np.zeros(N)

        for i in range(N):
            r = log_returns[:, i]
            mean_r = np.mean(r)
            var_r = np.var(r, ddof=1)
            std_r = np.sqrt(var_r)

            if std_r == 0:
                jumps = np.array([], dtype=float)
            else:
                jumps = r[np.abs(r - mean_r) > jump_threshold * std_r]

            self.lam[i] = len(jumps) / total_time

            if len(jumps) > 0:
                self.mu_j[i] = np.mean(jumps)
                self.sigma_j[i] = np.std(jumps, ddof=1) if len(jumps) > 1 else 0.0
            else:
                self.mu_j[i] = 0.0
                self.sigma_j[i] = 0.0

            jump_contribution_per_unit_time = self.lam[i] * (self.mu_j[i]**2 + self.sigma_j[i]**2)
            diffusion_var = var_r / self.delta_t - jump_contribution_per_unit_time
            diffusion_var = max(0.0, diffusion_var)
            self.sigma[i] = np.sqrt(diffusion_var)

            self.mu[i] = (mean_r / self.delta_t) - self.lam[i] * self.mu_j[i]

    def generate(self, num_samples, seq_length, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        N = self.mu.shape[0]
        l = seq_length if seq_length is not None else self.length
        R = np.zeros((num_samples, l, N))

        for i in range(N):
            mu = float(self.mu[i])
            sigma = float(self.sigma[i])
            lam = float(self.lam[i])
            mu_j = float(self.mu_j[i])
            sigma_j = float(self.sigma_j[i])
            pois_lambda = lam * self.delta_t

            for t in range(l):
                Z = np.random.randn(num_samples)
                drift = mu * self.delta_t
                diffusion = sigma * np.sqrt(self.delta_t) * Z
                N_jump = np.random.poisson(pois_lambda, size=num_samples)
                jump_sum = np.zeros(num_samples)
                idx_nonzero = np.nonzero(N_jump)[0]
                if idx_nonzero.size > 0 and (sigma_j > 0.0 or mu_j != 0.0):
                    for idx in idx_nonzero:
                        n = N_jump[idx]
                        jump_sum[idx] = np.sum(np.random.normal(mu_j, sigma_j, size=n))
                R[:, t, i] = drift + diffusion + jump_sum
        return R
