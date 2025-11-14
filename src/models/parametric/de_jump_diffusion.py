import torch
import numpy as np
from src.models.base.base_model import ParametricModel

class DoubleExponentialJumpDiffusion(ParametricModel):
    def __init__(self):
        super().__init__()
        self.mu = None
        self.sigma = None
        self.lam = None
        self.p = None
        self.eta1 = None
        self.eta2 = None
        self.kappa = None

    def fit(self, log_returns: torch.Tensor) -> None:
        log_returns = log_returns.flatten()
        T = log_returns.shape[0]
        jump_threshold = 3

        abs_median = torch.median(torch.abs(log_returns))
        small_mask = torch.abs(log_returns) < jump_threshold * abs_median
        diffusion_returns = log_returns[small_mask]
        self.sigma = torch.std(diffusion_returns, unbiased=True)

        jump_mask = ~small_mask
        jumps = log_returns[jump_mask]
        self.lam = jumps.shape[0] / T

        pos_jumps = jumps[jumps > 0]
        neg_jumps = jumps[jumps < 0]

        self.p = float(pos_jumps.shape[0] / max(jumps.shape[0], 1))
        self.eta1 = float(1.0 / pos_jumps.mean()) if pos_jumps.shape[0] > 0 else 1.0
        self.eta2 = float(-1.0 / neg_jumps.mean()) if neg_jumps.shape[0] > 0 else 1.0
        
        self.kappa = (self.p * self.eta1 / (self.eta1 - 1) if self.eta1 > 1 else 0.0) + \
                        ((1 - self.p) * self.eta2 / (self.eta2 + 1)) 
        mean_return = torch.mean(log_returns)
        self.mu = float(mean_return + 0.5 * self.sigma**2 + self.kappa * self.lam)
        print(f"mu: {self.mu}, sigma: {self.sigma}, lam: {self.lam}, p: {self.p}, eta1: {self.eta1}, eta2: {self.eta2}, kappa: {self.kappa}")

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        log_returns = torch.zeros((num_samples, generation_length))
        drift = (self.mu - 0.5 * self.sigma**2 - self.kappa * self.lam)
        diffusion = self.sigma * torch.randn(num_samples, generation_length)

        num_jumps = torch.poisson(torch.full((num_samples, generation_length), self.lam))
        jumps = torch.zeros_like(diffusion)

        for pos in [True, False]:
            mask = torch.rand_like(jumps) < self.p if pos else torch.rand_like(jumps) >= self.p
            rand_vals = torch.rand_like(jumps)
            if pos:
                jump_sizes = -torch.log(1 - rand_vals) / self.eta1
            else:
                jump_sizes = torch.log(rand_vals) / self.eta2
            jumps += num_jumps * mask.float() * jump_sizes

        log_returns = drift + diffusion + jumps
        return log_returns