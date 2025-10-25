import torch
import numpy as np
from typing import Optional

from src.models.base.base_model import ParametricModel


class OrnsteinUhlenbeckProcess(ParametricModel):
    """
    Ornstein-Uhlenbeck (O-U) parametric model for multichannel time series.

    Assumptions:
      - Input arrays shaped (l, N) where all N channels are log returns (already preprocessed).
      - No internal log return computation (data is already log returns).
      - All channels are feature signals.
      - No timestamp channel in input data.
      - Each channel is an independent O-U process.
      - Time steps are evenly spaced (linear) with constant dt = 1.
      - O-U dynamics: dX_t = theta * (mu - X_t) dt + sigma dW_t
        where X_t represents log returns
      - Estimation uses MLE for constant dt:
            X_{t+dt} = X_t * exp(-theta*dt) + mu*(1 - exp(-theta*dt)) + epsilon,
            epsilon ~ N(0, sigma^2*(1 - exp(-2*theta*dt))/(2*theta))
            
    Model Parameters:
      - mu: Long-term mean (mean reversion level, typically close to 0 for log returns)
      - theta: Mean reversion rate (speed of reversion)
      - sigma: Volatility
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
        self.theta = None
        self.sigma = None

        self.fitted_data = None

    def fit(self, data):
        """
        Fit the Ornstein-Uhlenbeck (O-U) process model parameters to `data`.

        Estimates per-channel parameters via MLE for evenly spaced time series:
            mu: Long-term mean.
            theta: Mean reversion rate.
            sigma: Volatility.
            
        Data is assumed to be log returns already.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2D array with shape (l, N)")

        l, N = data.shape
        if N != self.num_channels:
            raise ValueError(f"data has {N} channels but model expects {self.num_channels}")

        self.fitted_data = data.copy()

        mu = np.zeros(self.num_channels, dtype=np.float64)
        theta = np.zeros(self.num_channels, dtype=np.float64)
        sigma = np.zeros(self.num_channels, dtype=np.float64)

        for ch in range(self.num_channels):
            series = data[:, ch].astype(np.float64)
            if len(series) < 2:
                mu[ch] = 0.0
                theta[ch] = 0.1
                sigma[ch] = 1e-6
                continue
            
            X_prev = series[:-1]
            X_next = series[1:]
            
            # AR(1) estimate
            cov = np.cov(X_prev, X_next, bias=True)
            var_x = cov[0, 0]
            cov_xy = cov[0, 1]
            
            phi = cov_xy / var_x if var_x > 1e-10 else 0.9
            phi = np.clip(phi, 1e-8, 0.9999)  # positive and <1
            
            # Continuous-time parameters assuming dt = 1
            theta_hat = -np.log(phi)
            theta_hat = max(theta_hat, 1e-6)
            
            mu_hat = np.mean(X_next - phi * X_prev) / (1 - phi) if abs(1 - phi) > 1e-10 else np.mean(series)
            
            # Residuals
            eps = X_next - (phi * X_prev + (1 - phi) * mu_hat)
            var_eps = np.var(eps, ddof=0)
            var_eps = max(var_eps, 1e-10)
            
            # Sigma estimation
            sigma_hat = np.sqrt(2 * theta_hat * var_eps / (1 - np.exp(-2 * theta_hat)))
            sigma_hat = max(sigma_hat, 1e-6)
            
            mu[ch] = mu_hat
            theta[ch] = theta_hat
            sigma[ch] = sigma_hat
        
        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)

        return {"mu": self.mu, "theta": self.theta, "sigma": self.sigma}

    def generate(self, num_samples: int, output_length: Optional[int] = None, 
                 seed: Optional[int] = None):
        """
        Generate `num_samples` independent sample paths of log returns
        for the Ornstein-Uhlenbeck (O-U) process.

        Returns a torch tensor of shape (num_samples, output_length, num_channels).
        All channels are simulated log return channels.

        O-U assumptions:
        - Mean-reverting process: dX_t = theta * (mu - X_t) dt + sigma dW_t
        - Multi-channel support
        - Each time step generates a log return value
        
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

        mu = self.mu.to(device) if self.mu is not None else torch.zeros(N, device=device)
        theta = self.theta.to(device) if self.theta is not None else torch.full((N,), 0.1, device=device)
        sigma = self.sigma.to(device) if self.sigma is not None else torch.ones(N, device=device) * 1e-3

        # Initialize at the long-term mean
        if hasattr(self, "fitted_data") and self.fitted_data is not None:
            init_vals = torch.tensor(self.fitted_data[0, :], dtype=torch.float32, device=device)
        else:
            init_vals = mu.clone()
        
        paths[:, 0, :] = init_vals.unsqueeze(0).expand(num_samples, -1)

        # Generate O-U process
        for k in range(L - 1):
            z = torch.randn((num_samples, N), dtype=torch.float32, device=device)
            prev = paths[:, k, :]

            exp_neg_theta = torch.exp(-theta)
            mean = mu * (1 - exp_neg_theta) + prev * exp_neg_theta
            std = sigma * torch.sqrt((1 - exp_neg_theta ** 2) / (2 * theta + 1e-8))
            paths[:, k + 1, :] = mean + std * z

        return paths
