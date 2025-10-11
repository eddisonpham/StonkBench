import torch
import numpy as np
from src.models.base.base_model import ParametricModel

class GeometricBrownianMotion(ParametricModel):
    """
    Geometric Brownian Motion (GBM)
    dS = μS dt + σS dW
    """

    def __init__(self, length, num_channels, initial_price=1.0):
        super().__init__()
        self.length = length
        self.num_channels = num_channels
        self.initial_price = initial_price
        
        self.mu = torch.zeros(num_channels, device=self.device)
        self.sigma = torch.ones(num_channels, device=self.device) * 0.2

    def fit(self, data_loader):
        """Fit GBM parameters from price data"""
        all_prices = []
        for batch in data_loader:
            if batch.dim() == 2:
                batch = batch.unsqueeze(0)
            all_prices.append(batch.cpu().numpy())
        
        prices = np.concatenate(all_prices, axis=0)  # (R, l, N)
        
        for channel in range(self.num_channels):
            price_series = prices[:, :, channel].flatten()
            if len(price_series) < 2:
                continue
            
            log_returns = np.diff(np.log(price_series + 1e-8))
            if len(log_returns) > 0:
                self.mu[channel] = float(np.mean(log_returns))
                self.sigma[channel] = float(np.std(log_returns))
                self.sigma[channel] = torch.clamp(self.sigma[channel], 0.01, 1.0)

    def generate(self, num_samples):
        """Generate multivariate GBM time series"""
        dt = 1.0 / self.length

        # Initialize price tensor
        prices = torch.zeros(num_samples, self.length, self.num_channels, device=self.device)
        prices[:, 0, :] = self.initial_price

        for t in range(1, self.length):
            # Brownian increments
            Z = torch.randn(num_samples, self.num_channels, device=self.device)
            
            # Apply GBM evolution: S_t = S_{t-1} * exp((μ - 0.5σ²)dt + σ√dt * Z)
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * torch.sqrt(torch.tensor(dt, device=self.device)) * Z
            prices[:, t, :] = prices[:, t - 1, :] * torch.exp(drift + diffusion)
        
        return prices
