import torch
import numpy as np
from scipy import stats
from src.models.base.base_model import ParametricModel

class OrnsteinUhlenbeckProcess(ParametricModel):
    """
    Ornstein-Uhlenbeck Process.
    
    The OU process follows: dX = θ(μ - X)dt + σdW
    Where:
    - θ: mean reversion speed
    - μ: long-term mean
    - σ: volatility
    - W: Wiener process
    """
    
    def __init__(self, length, num_channels, dt=0.01):
        super().__init__()
        self.length = length
        self.num_channels = num_channels
        self.dt = dt
        
        # Initialize parameters with reasonable defaults
        self.theta = torch.ones(num_channels, device=self.device) * 0.1  # mean reversion
        self.mu = torch.zeros(num_channels, device=self.device)           # long-term mean
        self.sigma = torch.ones(num_channels, device=self.device) * 0.1   # volatility

    def fit(self, data_loader):
        """Fit OU parameters using maximum likelihood estimation"""
        # Collect all data
        all_data = []
        for batch in data_loader:
            if batch.dim() == 2:
                batch = batch.unsqueeze(0)
            all_data.append(batch.cpu().numpy())
        
        data = np.concatenate(all_data, axis=0)  # Shape: (R, l, N)
        
        for channel in range(self.num_channels):
            # Flatten time series for this channel
            time_series = data[:, :, channel].flatten()
            
            if len(time_series) < 2:
                continue
                
            # Use numpy for simpler parameter estimation
            returns = np.diff(time_series)
            
            # Simple parameter estimation
            self.mu[channel] = float(np.mean(time_series))
            self.sigma[channel] = float(np.std(returns))
            
            # Estimate mean reversion speed using autocorrelation
            if len(returns) > 1:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                if not np.isnan(autocorr):
                    # Convert autocorrelation to mean reversion speed
                    self.theta[channel] = float(max(0.01, -np.log(max(0.01, autocorr)) / self.dt))
            
            # Ensure parameters are reasonable
            self.theta[channel] = torch.clamp(self.theta[channel], 0.01, 10.0)
            self.sigma[channel] = torch.clamp(self.sigma[channel], 0.01, 1.0)

    def generate(self, num_samples):
        """Generate OU process paths using vectorized operations"""
        # Initialize output tensor
        paths = torch.zeros(num_samples, self.length, self.num_channels, device=self.device)
        
        # Start from long-term mean with small random perturbation
        current_value = self.mu.unsqueeze(0).expand(num_samples, -1)
        current_value += 0.01 * torch.randn(num_samples, self.num_channels, device=self.device)
        paths[:, 0, :] = current_value
        
        # Generate paths using Euler-Maruyama method (vectorized)
        for t in range(1, self.length):
            # Random shocks
            dW = torch.randn(num_samples, self.num_channels, device=self.device) * np.sqrt(self.dt)
            
            # OU process update: dX = θ(μ - X)dt + σdW
            mean_reversion = self.theta * (self.mu - current_value) * self.dt
            diffusion = self.sigma * dW
            
            current_value += mean_reversion + diffusion
            paths[:, t, :] = current_value
            
        return paths
