import torch
import numpy as np
from typing import Optional

from src.models.base.base_model import ParametricModel


class GeometricBrownianMotion(ParametricModel):
    """
    Geometric Brownian Motion parametric model for multichannel time series.

    Assumptions:
      - Input arrays shaped (l, N) where all N channels are log returns (already preprocessed).
      - No internal log return computation (data is already log returns).
      - All channels are feature signals.
      - No timestamp channel in input data.
      - Each channel is treated as an independent GBM.
      - Time steps are evenly spaced (linear). The model uses dt = 1 as the unit time step.
      - Log returns are modeled as: r_t ~ N(mu - 0.5*sigma^2, sigma^2).
    """
    def __init__(
        self,
        length: int,
        num_channels: int,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.length = int(length)
        self.num_channels = int(num_channels)
        self.device = device if device is not None else torch.device("cpu")

        self.mu = None
        self.sigma = None

        self.fitted_data = None

    def fit(self, data):
        """
        Fit GBM parameters to `data` (numpy array or torch tensor) of shape (l, N).
        
        Data is assumed to be log returns, so we estimate:
            mu: drift parameter (annualized log return)
            sigma: volatility parameter (annualized volatility)
        
        Estimation:
            mu_hat = mean(returns) + 0.5 * var(returns)
            sigma_hat = std(returns)
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2D array with shape (l, N)")

        self.fitted_data = data.copy()

        mu = np.zeros(self.num_channels, dtype=np.float64)
        sigma = np.zeros(self.num_channels, dtype=np.float64)

        for ch in range(self.num_channels):
            returns = data[:, ch].astype(np.float64)
            
            if len(returns) < 2:
                mu[ch] = 0.0
                sigma[ch] = 1e-6
                continue

            # Estimate parameters from log returns
            mean_return = float(np.mean(returns))
            std_return = float(np.std(returns, ddof=1))
            
            # Ensure numerical stability
            std_return = max(std_return, 1e-6)
            
            # GBM parameters
            sigma[ch] = std_return
            mu[ch] = mean_return + 0.5 * std_return**2

        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        
        return {"mu": self.mu, "sigma": self.sigma}

    def generate(self, num_samples: int, output_length: Optional[int] = None, 
                 seed: Optional[int] = None):
        """
        Generate `num_samples` independent sample paths of log returns.

        Returns a torch tensor of shape (num_samples, output_length, num_channels).
        All channels are simulated log return channels.

        Args:
            num_samples (int): Number of samples to generate.
            output_length (int, optional): Length of generated sequences. Defaults to self.length.
            seed (int, optional): Random seed for reproducibility.
            
        Returns:
            torch.Tensor: Generated log returns of shape (num_samples, output_length, num_channels)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = self.device
        L = int(output_length) if output_length is not None else int(self.length)
        N = self.num_channels

        paths = torch.zeros((num_samples, L, N), dtype=torch.float32, device=device)

        if self.mu is None or self.sigma is None:
            mu = torch.zeros((N,), dtype=torch.float32, device=device)
            sigma = torch.ones((N,), dtype=torch.float32, device=device) * 1e-6
        else:
            mu = self.mu.to(device)
            sigma = self.sigma.to(device)

        # Generate log returns directly
        for k in range(L):
            z = torch.randn((num_samples, N), dtype=torch.float32, device=device)
            drift = mu - 0.5 * sigma * sigma
            diffusion = sigma * z
            paths[:, k, :] = drift + diffusion

        return paths
