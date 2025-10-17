import torch
import numpy as np
from src.models.base.base_model import ParametricModel

class DoubleExponentialJumpDiffusion(ParametricModel):
    """
    Double Exponential Jump Diffusion Model (Kou Model)
    dS = μS dt + σS dW + S∫(e^Y - 1)N(dt,dy)
    where Y has double exponential distribution
    """

    def __init__(self, length, num_channels, initial_price=1.0):
        super().__init__()
        self.length = length
        self.num_channels = num_channels
        self.initial_price = initial_price
        
        # Diffusion parameters
        self.mu = torch.zeros(num_channels, device=self.device)
        self.sigma = torch.ones(num_channels, device=self.device) * 0.2
        
        # Jump parameters
        self.lambda_jump = torch.ones(num_channels, device=self.device) * 0.1  # Jump intensity
        self.p = torch.ones(num_channels, device=self.device) * 0.5  # Probability of upward jump
        self.eta1 = torch.ones(num_channels, device=self.device) * 0.1  # Upward jump rate
        self.eta2 = torch.ones(num_channels, device=self.device) * 0.1  # Downward jump rate

    def fit(self, data_loader):
        """Fit Double Exponential Jump Diffusion parameters from price data"""
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
                # Basic parameter estimation
                self.mu[channel] = float(np.mean(log_returns))
                
                # Separate normal and extreme returns
                threshold = 2 * np.std(log_returns)
                normal_returns = log_returns[np.abs(log_returns) <= threshold]
                extreme_returns = log_returns[np.abs(log_returns) > threshold]
                
                if len(normal_returns) > 0:
                    self.sigma[channel] = float(np.std(normal_returns))
                    self.sigma[channel] = torch.clamp(self.sigma[channel], 0.01, 1.0)
                
                if len(extreme_returns) > 0:
                    # Estimate jump parameters
                    self.lambda_jump[channel] = len(extreme_returns) / len(log_returns)
                    
                    positive_jumps = extreme_returns[extreme_returns > 0]
                    negative_jumps = extreme_returns[extreme_returns < 0]
                    
                    if len(positive_jumps) > 0 and len(negative_jumps) > 0:
                        self.p[channel] = len(positive_jumps) / len(extreme_returns)
                        self.eta1[channel] = 1.0 / np.mean(positive_jumps) if np.mean(positive_jumps) > 0 else 10.0
                        self.eta2[channel] = -1.0 / np.mean(negative_jumps) if np.mean(negative_jumps) < 0 else 10.0
                    else:
                        self.p[channel] = 0.5
                        self.eta1[channel] = 10.0
                        self.eta2[channel] = 10.0

    def _sample_double_exponential(self, size, p, eta1, eta2):
        """Sample from double exponential distribution"""
        uniform = torch.rand(size, device=self.device)
        exponential1 = torch.exponential(torch.ones(size, device=self.device))
        exponential2 = torch.exponential(torch.ones(size, device=self.device))
        
        # Choose upward or downward jump
        upward_mask = uniform < p
        
        jumps = torch.zeros(size, device=self.device)
        jumps[upward_mask] = exponential1[upward_mask] / eta1
        jumps[~upward_mask] = -exponential2[~upward_mask] / eta2
        
        return jumps

    def generate(self, num_samples, linear_timestamps=True, output_length=None, seed=None):
        """Generate multivariate Double Exponential Jump Diffusion time series"""
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
            jump_indicators = torch.poisson(self.lambda_jump * dt)
            total_jump = torch.zeros(num_samples, self.num_channels, device=self.device)
            
            for channel in range(self.num_channels):
                for sample in range(num_samples):
                    n_jumps = int(jump_indicators[channel].item())
                    if n_jumps > 0:
                        jump_sizes = self._sample_double_exponential(
                            n_jumps, 
                            self.p[channel], 
                            self.eta1[channel], 
                            self.eta2[channel]
                        )
                        total_jump[sample, channel] = torch.sum(jump_sizes)
            
            # Apply evolution equation
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * torch.sqrt(torch.tensor(dt, device=self.device)) * Z
            
            prices[:, t, :] = prices[:, t - 1, :] * torch.exp(drift + diffusion + total_jump)
        
        return prices