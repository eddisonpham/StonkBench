import torch
import numpy as np
from src.models.base.base_model import ParametricModel

class GARCH11(ParametricModel):
    """
    GARCH(1,1) Model
    r_t = μ + σ_t * ε_t
    σ_t² = ω + α * r_{t-1}² + β * σ_{t-1}²
    where ε_t ~ N(0,1)
    """

    def __init__(self, length, num_channels, initial_price=1.0):
        super().__init__()
        self.length = length
        self.num_channels = num_channels
        self.initial_price = initial_price
        
        # Mean equation parameters
        self.mu = torch.zeros(num_channels, device=self.device)
        
        # GARCH parameters
        self.omega = torch.ones(num_channels, device=self.device) * 0.01  # Constant
        self.alpha = torch.ones(num_channels, device=self.device) * 0.1   # ARCH term
        self.beta = torch.ones(num_channels, device=self.device) * 0.8    # GARCH term
        
        # Initial conditional variance
        self.sigma2_0 = torch.ones(num_channels, device=self.device) * 0.01

    def fit(self, data_loader):
        """Fit GARCH(1,1) parameters from return data"""
        all_data = []
        for batch in data_loader:
            if batch.dim() == 2:
                batch = batch.unsqueeze(0)
            all_data.append(batch.cpu().numpy())
        
        data = np.concatenate(all_data, axis=0)  # (R, l, N)
        
        for channel in range(self.num_channels):
            series = data[:, :, channel].flatten()
            if len(series) < 2:
                continue
            
            # Convert prices to returns if needed
            if np.all(series > 0):  # Assume these are prices
                returns = np.diff(np.log(series + 1e-8))
            else:  # Assume these are already returns
                returns = series[1:]
            
            if len(returns) > 10:
                # Simple parameter estimation (can be improved with MLE)
                self.mu[channel] = float(np.mean(returns))
                
                # Estimate GARCH parameters using method of moments
                returns_centered = returns - np.mean(returns)
                var_unconditional = np.var(returns_centered)
                
                # Simple initialization
                self.omega[channel] = var_unconditional * 0.1
                self.alpha[channel] = 0.1
                self.beta[channel] = 0.8
                
                # Ensure stationarity constraint: alpha + beta < 1
                sum_params = self.alpha[channel] + self.beta[channel]
                if sum_params >= 1.0:
                    self.alpha[channel] = 0.1
                    self.beta[channel] = 0.85
                
                self.sigma2_0[channel] = var_unconditional

    def generate(self, num_samples, linear_timestamps=True, output_length=None, seed=None):
        """Generate multivariate GARCH(1,1) time series"""
        if seed is not None:
            torch.manual_seed(seed)
            
        if output_length is not None:
            length = output_length
        else:
            length = self.length

        # Initialize return and variance tensors
        returns = torch.zeros(num_samples, length, self.num_channels, device=self.device)
        sigma2 = torch.zeros(num_samples, length, self.num_channels, device=self.device)
        
        # Initial conditions
        sigma2[:, 0, :] = self.sigma2_0
        
        # Generate innovations
        epsilon = torch.randn(num_samples, length, self.num_channels, device=self.device)

        for t in range(length):
            # Generate returns
            returns[:, t, :] = self.mu + torch.sqrt(sigma2[:, t, :]) * epsilon[:, t, :]
            
            # Update conditional variance for next period
            if t < length - 1:
                sigma2[:, t + 1, :] = (
                    self.omega + 
                    self.alpha * returns[:, t, :] ** 2 + 
                    self.beta * sigma2[:, t, :]
                )
        
        # Convert returns to prices
        prices = torch.zeros(num_samples, length, self.num_channels, device=self.device)
        prices[:, 0, :] = self.initial_price
        
        for t in range(1, length):
            prices[:, t, :] = prices[:, t - 1, :] * torch.exp(returns[:, t, :])
        
        return prices