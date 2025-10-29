import torch
import numpy as np
from src.models.base.base_model import ParametricModel

class DoubleExponentialJumpDiffusion(ParametricModel):
    def __init__(self, length, num_channels, device='cpu'):
        super().__init__(length, num_channels)
        self.device = device
        self.mu = None
        self.sigma = None
        self.lam = None
        self.p = None
        self.eta1 = None
        self.eta2 = None

    def fit(self, log_returns):
        """
        Fit DEJD parameters to log_returns (assumes uniform time steps)
        :param log_returns: (L x N) numpy array or torch tensor
        """
        log_returns = torch.tensor(log_returns, device=self.device)
        L, N = log_returns.shape

        self.mu = torch.zeros(N, device=self.device)
        self.sigma = torch.zeros(N, device=self.device)
        self.lam = torch.zeros(N, device=self.device)
        self.p = torch.zeros(N, device=self.device)
        self.eta1 = torch.zeros(N, device=self.device)
        self.eta2 = torch.zeros(N, device=self.device)

        for i in range(N):
            r = log_returns[:, i]
            med = r.median()
            mad = (r - med).abs().median()
            threshold = 5 * mad

            jumps = r[torch.abs(r - med) > threshold]
            num_jumps = len(jumps)
            self.lam[i] = num_jumps / L if L > 0 else 0.0

            pos_jumps = jumps[jumps > 0]
            neg_jumps = jumps[jumps < 0]

            self.p[i] = len(pos_jumps) / num_jumps if num_jumps > 0 else 0.5
            self.eta1[i] = 1.0 / (pos_jumps.mean() + 1e-8) if len(pos_jumps) > 0 else 1.0
            self.eta2[i] = 1.0 / (-neg_jumps.mean() + 1e-8) if len(neg_jumps) > 0 else 1.0

            # Expected jump moments
            EY = self.p[i] / self.eta1[i] - (1.0 - self.p[i]) / self.eta2[i]
            EY2 = 2.0 * (self.p[i] / self.eta1[i]**2 + (1.0 - self.p[i]) / self.eta2[i]**2)

            mean_r = r.mean()
            var_r = r.var(unbiased=True)
            self.mu[i] = mean_r - self.lam[i] * EY
            self.sigma[i] = torch.sqrt(torch.clamp(var_r - self.lam[i] * EY2, min=1e-8))

    def generate(self, num_samples, seq_length=None, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        N = self.mu.shape[0]
        L = seq_length if seq_length is not None else self.length

        # Broadcast mu, sigma, lam for vectorized operations
        mu_vec = self.mu.unsqueeze(0)
        sigma_vec = self.sigma.unsqueeze(0)
        lam_vec = self.lam.unsqueeze(0)

        log_returns = torch.zeros((num_samples, L, N), device=self.device)

        for t in range(L):
            eps = torch.randn((num_samples, N), device=self.device)
            diffusion = eps * sigma_vec
            drift = mu_vec

            # Poisson jumps
            num_jumps = torch.poisson(lam_vec.expand(num_samples, -1))
            jumps = torch.zeros((num_samples, N), device=self.device)

            for i in range(N):
                nj = num_jumps[:, i].long()
                mask = nj > 0
                if mask.any():
                    total_jumps = nj[mask].sum().item()
                    # Asymmetric double exponential sampling
                    u = torch.rand(total_jumps, device=self.device)
                    v = torch.rand(total_jumps, device=self.device)
                    jump_vals = torch.zeros(total_jumps, device=self.device)
                    pos_mask = u <= self.p[i]
                    jump_vals[pos_mask] = -torch.log(v[pos_mask] + 1e-12) / self.eta1[i]
                    jump_vals[~pos_mask] = -(-torch.log(v[~pos_mask] + 1e-12) / self.eta2[i])

                    # assign jumps per sample
                    idx = 0
                    for j in mask.nonzero(as_tuple=False).squeeze(1):
                        count = nj[j].item()
                        if count > 0:
                            jumps[j, i] = jump_vals[idx:idx+count].sum()
                            idx += count

            log_returns[:, t, :] = drift + diffusion + jumps

        return log_returns.cpu()
