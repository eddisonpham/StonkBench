import torch
import numpy as np
from arch import arch_model

from src.models.base.base_model import ParametricModel


class GARCH11(ParametricModel):
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.mu = None
        self.omega = None
        self.alpha = None
        self.beta = None

    def fit(self, log_returns: torch.Tensor) -> None:
        log_returns_np = log_returns.detach().cpu().numpy()
        am = arch_model(
            log_returns_np,
            mean='Constant',
            vol='GARCH',
            p=1,
            q=1,
            dist='normal',
            rescale=False
        )
        model_fit = am.fit(disp='off')
        self.mu = model_fit.params['mu']
        self.omega = model_fit.params['omega']
        self.alpha = model_fit.params['alpha[1]']
        self.beta = model_fit.params['beta[1]']

    def generate(self, num_samples: int, generation_length: int) -> torch.Tensor:
        log_returns = torch.zeros((num_samples, generation_length))
        sigma2 = torch.zeros((num_samples, generation_length))
        epsilon = torch.zeros((num_samples, generation_length))
        sigma2[:, 0] = self.omega / (1 - self.alpha - self.beta)
        epsilon[:, 0] = torch.sqrt(sigma2[:, 0]) * torch.randn(num_samples)
        log_returns[:, 0] = self.mu + epsilon[:, 0]
        for t in range(1, generation_length):
            sigma2[:, t] = self.omega + self.alpha * epsilon[:, t-1]**2 + self.beta * sigma2[:, t-1]
            epsilon[:, t] = torch.sqrt(sigma2[:, t]) * torch.randn(num_samples)
            log_returns[:, t] = self.mu + epsilon[:, t]
        return log_returns
