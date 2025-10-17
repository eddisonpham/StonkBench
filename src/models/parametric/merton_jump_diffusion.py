import torch
import numpy as np
from src.models.base.base_model import ParametricModel

class MertonJumpDiffusion(ParametricModel):
    """
    Merton Jump Diffusion Model
    dS = μS dt + σS dW + (J-1)S dN
    where J ~ log-normal jumps and dN is a Poisson process
    """

    def __init__(self, length, num_channels, initial_price=1.0):
        super().__init__()
        self.length = length
        self.num_channels = num_channels
        self.initial_price = initial_price
        
        # Drift and volatility parameters
        self.mu = torch.zeros(num_channels, device=self.device)
        self.sigma = torch.ones(num_channels, device=self.device) * 0.2
        
        # Jump parameters
        self.lambda_jump = torch.ones(num_channels, device=self.device) * 0.1  # Jump intensity
        self.mu_jump = torch.zeros(num_channels, device=self.device)  # Jump size mean
        self.sigma_jump = torch.ones(num_channels, device=self.device) * 0.1  # Jump size std

    def fit(self, data_loader):
        """Fit Merton Jump Diffusion parameters from price data"""
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
                # Simple parameter estimation (can be improved with MLE)
                self.mu[channel] = float(np.mean(log_returns))
                self.sigma[channel] = float(np.std(log_returns)) * 0.8  # Reduce continuous vol
                self.sigma[channel] = torch.clamp(self.sigma[channel], 0.01, 1.0)
                
                # Estimate jump parameters (simple approach)
                extreme_returns = log_returns[np.abs(log_returns) > 2 * np.std(log_returns)]
                if len(extreme_returns) > 0:
                    self.lambda_jump[channel] = len(extreme_returns) / len(log_returns)
                    self.mu_jump[channel] = float(np.mean(extreme_returns))
                    self.sigma_jump[channel] = float(np.std(extreme_returns)) if len(extreme_returns) > 1 else 0.1

    def generate(self, num_samples, linear_timestamps=True, output_length=None, seed=None):
        """Generate multivariate Merton Jump Diffusion time series"""
        if seed is not None:
            torch.manual_seed(seed)
            
        if output_length is not None:
            length = output_length
        else:
            length = self.length
            
        dt = 1.0 / length

        # Initialize price tensor
        prices = torch.zeros(num_samples, length, self.num_channels, device=self.device)
        prices[:, 0, :] = self.initial_price

        for t in range(1, length):
            # Brownian increments
            Z = torch.randn(num_samples, self.num_channels, device=self.device)
            
            # Jump process
            jump_times = torch.poisson(self.lambda_jump * dt)  # Number of jumps
            jump_sizes = torch.zeros(num_samples, self.num_channels, device=self.device)
            
            for channel in range(self.num_channels):
                for sample in range(num_samples):
                    n_jumps = int(jump_times[channel].item())
                    if n_jumps > 0:
                        # Log-normal jump sizes
                        jump_magnitudes = torch.normal(
                            self.mu_jump[channel], 
                            self.sigma_jump[channel], 
                            size=(n_jumps,)
                        )
                        jump_sizes[sample, channel] = torch.sum(jump_magnitudes)
            
            # Apply Merton evolution
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * torch.sqrt(torch.tensor(dt, device=self.device)) * Z
            
            prices[:, t, :] = prices[:, t - 1, :] * torch.exp(drift + diffusion + jump_sizes)
        
        return prices