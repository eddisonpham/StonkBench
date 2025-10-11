import torch
from src.models.base.base_model import ParametricModel

class GeometricBrownianMotion(ParametricModel):
    def __init__(self, length, num_channels, initial_price=1.0):
        super().__init__()
        self.length = length
        self.num_channels = num_channels
        self.initial_price = initial_price
        self.mu = torch.zeros(num_channels, device=self.device)
        self.sigma = torch.ones(num_channels, device=self.device)

    def fit(self, data_loader):
        log_returns = []
        for batch in data_loader:
            # Ensure batch is (R, l, N)
            if batch.dim() == 2:
                batch = batch.unsqueeze(0) # Add R dimension if missing
            prices = batch.permute(0, 2, 1) # (R, N, l)
            
            log_prices = torch.log(prices + 1e-8)
            log_returns.append(log_prices[:, :, 1:] - log_prices[:, :, :-1])
        
        log_returns = torch.cat(log_returns, dim=0)
        self.mu = torch.mean(log_returns, dim=(0, 2)).to(self.device)
        self.sigma = torch.std(log_returns, dim=(0, 2)).to(self.device)

    def generate(self, num_samples):
        dt = 1.0 / self.length
        prices = torch.zeros(num_samples, self.num_channels, self.length, device=self.device)
        prices[:, :, 0] = self.initial_price

        for t in range(1, self.length):
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * torch.sqrt(torch.tensor(dt, device=self.device)) * torch.randn(num_samples, self.num_channels, device=self.device)
            prices[:, :, t] = prices[:, :, t-1] * torch.exp(drift + diffusion)
        
        return prices.permute(0, 2, 1) # (num_samples, length, channels)
