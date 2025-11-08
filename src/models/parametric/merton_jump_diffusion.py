import torch
import numpy as np

from src.models.base.base_model import ParametricModel


class MertonJumpDiffusion(ParametricModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.mu = None
        self.sigma = None
        self.lam = None
        self.mu_j = None
        self.sigma_j = None
        self.kappa = None

    def fit(self, log_returns: torch.Tensor) -> None:
        log_returns_np = log_returns.detach().cpu().numpy()
        self.sigma = np.std(log_returns_np)
        
        threshold = 3.0 * self.sigma
        jumps = np.where(np.abs(log_returns_np) > threshold)
        num_jumps = len(jumps[0])
        total_obs = len(log_returns_np)

        self.lam = num_jumps / total_obs

        if num_jumps > 0:
            jumps = log_returns_np[jumps]
            self.mu_j = np.mean(jumps)
            self.sigma_j = np.std(jumps)
        else:
            self.mu_j = 0.0
            self.sigma_j = 0.0

        self.kappa = torch.exp(torch.tensor(self.mu_j + 0.5 * self.sigma_j**2))-1
        self.mu = np.mean(log_returns_np) + 0.5 * self.sigma**2 + self.kappa * self.lam
        print(f"mu: {self.mu}, sigma: {self.sigma}, kappa: {self.kappa}, lam: {self.lam}, mu_j: {self.mu_j}, sigma_j: {self.sigma_j}")

    def generate(self, num_samples: int, generation_length: int) -> torch.Tensor:
        log_returns = torch.zeros((num_samples, generation_length))
        for t in range(generation_length):
            epsilon = torch.randn(num_samples)
            diffusion = (self.mu - 0.5 * self.sigma**2 - self.lam * self.kappa) + self.sigma * epsilon
            N = torch.poisson(torch.full((num_samples,), self.lam))
            jumps = torch.zeros(num_samples)
            for i in range(num_samples):
                if N[i] > 0:
                    jumps[i] = torch.sum(self.mu_j + self.sigma_j * torch.randn(int(N[i].item())))
            log_returns[:, t] = diffusion + jumps
        return log_returns