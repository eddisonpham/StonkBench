import torch
from models.base_model import ParametricModel

class OrnsteinUhlenbeckProcess(ParametricModel):
    def __init__(self, length, num_channels, dt=0.01):
        super().__init__()
        self.length = length
        self.num_channels = num_channels
        self.dt = dt

        # Initialize parameters
        self.theta = 0.1 * torch.ones(num_channels, device=self.device)  # Mean reversion speed
        self.mu = torch.zeros(num_channels, device=self.device)          # Mean level
        self.sigma = 0.1 * torch.ones(num_channels, device=self.device)  # Volatility

    def fit(self, data_loader):
        """Estimate O-U parameters from time series data"""
        data_points = []
        for batch in data_loader:
            if batch.dim() == 2:
                batch = batch.unsqueeze(0)
            data_points.append(batch)
        data_points = torch.cat(data_points, dim=0)  # (R, L, N)

        for n in range(self.num_channels):
            X = data_points[:, :, n]  # (R, L)
            X = X.flatten()           # Flatten to 1D for estimation

            X_t = X[:-1]
            X_t1 = X[1:]

            # Compute mean and autocovariance
            mean_X = X.mean()
            var_X = X.var(unbiased=True)
            cov_X = ((X_t - mean_X) * (X_t1 - mean_X)).mean()

            # Estimate theta
            if var_X < 1e-8:
                # Nearly constant series, keep defaults
                continue
            phi = cov_X / var_X
            theta_hat = -torch.log(phi + 1e-8) / self.dt  # Ensure log arg >0

            # Estimate mu
            mu_hat = X.mean()

            # Estimate sigma
            residuals = X_t1 - X_t * phi
            sigma_hat = torch.sqrt(residuals.pow(2).mean() * 2 * theta_hat / (1 - phi**2 + 1e-8))

            # Clamp parameters to avoid explosion
            self.theta[n] = torch.clamp(theta_hat, min=1e-3, max=10.0)
            self.mu[n] = torch.clamp(mu_hat, min=-10.0, max=10.0)
            self.sigma[n] = torch.clamp(sigma_hat, min=1e-3, max=1.0)

    def generate(self, num_samples):
        """Generate O-U sample paths"""
        x = torch.zeros(num_samples, self.length, self.num_channels, device=self.device)

        # Initialize at mu
        current_x = self.mu + 0.01 * torch.randn(num_samples, self.num_channels, device=self.device)
        x[:, 0, :] = current_x

        for t in range(1, self.length):
            dw = torch.randn(num_samples, self.num_channels, device=self.device) * torch.sqrt(torch.tensor(self.dt, device=self.device))
            dx = self.theta * (self.mu - current_x) * self.dt + self.sigma * dw
            current_x = current_x + dx
            x[:, t, :] = current_x

        return x
