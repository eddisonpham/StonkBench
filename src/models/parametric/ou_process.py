import torch
import numpy as np
from typing import Optional

from src.models.base.base_model import ParametricModel


class OrnsteinUhlenbeckProcess(ParametricModel):
    """
    Ornstein-Uhlenbeck (O-U) parametric model for multichannel time series.

    Assumptions:
      - Input arrays shaped (l, N) where all N channels are feature signals.
      - No timestamp channel in input data.
      - Each channel is an independent O-U process.
      - Time steps are evenly spaced (linear) with constant dt.
      - Estimation uses MLE for constant dt:
            X_{t+dt} = X_t * exp(-theta*dt) + mu*(1 - exp(-theta*dt)) + epsilon,
            epsilon ~ N(0, sigma^2*(1 - exp(-2*theta*dt))/(2*theta))
    """

    def __init__(
        self,
        length: int,
        num_channels: int,
        initial_value: Optional[float] = 0.0, 
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.length = int(length)
        self.num_channels = int(num_channels)
        self.initial_value = initial_value
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
                mu[ch] = series[0] if len(series) > 0 else 0.0
                theta[ch] = 0.0
                sigma[ch] = 0.0
                continue
            
            X_prev = series[:-1]
            X_next = series[1:]
            
            # AR(1) estimate
            cov = np.cov(X_prev, X_next, bias=True)
            var_x = cov[0, 0]
            cov_xy = cov[0, 1]
            
            phi = cov_xy / var_x if var_x > 0 else 0.0
            phi = np.clip(phi, 1e-8, 0.999)  # positive and <1
            
            # Continuous-time parameters assuming dt = 1
            theta_hat = -np.log(phi)
            mu_hat = np.mean(X_next - phi * X_prev) / (1 - phi)
            
            # Residuals
            eps = X_next - (phi * X_prev + (1 - phi) * mu_hat)
            var_eps = np.var(eps, ddof=0)
            
            # Sigma estimation
            sigma_hat = np.sqrt(2 * theta_hat * var_eps / (1 - np.exp(-2 * theta_hat)))
            
            mu[ch] = mu_hat
            theta[ch] = theta_hat
            sigma[ch] = sigma_hat
        
        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)

        return {"mu": self.mu, "theta": self.theta, "sigma": self.sigma}

    def generate(self, num_samples: int, initial_value: Optional[np.ndarray] = None,
        output_length: Optional[int] = None, seed: Optional[int] = None
    ):
        """
        Generate `num_samples` independent sample paths of length `output_length` (or self.length if None)
        for the Ornstein-Uhlenbeck (O-U) process.

        Returns a torch tensor of shape (num_samples, output_length, num_channels).
        All channels are simulated feature channels.

        O-U assumptions:
        - Mean-reverting process: dX_t = theta * (mu - X_t) dt + sigma dW_t
        - Multi-channel support
        - Initial values can be provided, else fitted data or default is used
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = self.device
        L = int(output_length) if output_length is not None else int(self.length)
        N = self.num_channels

        paths = torch.zeros((num_samples, L, N), dtype=torch.float32, device=device)

        if initial_value is None:
            if hasattr(self, "fitted_data") and self.fitted_data is not None:
                init_vals = np.asarray(self.fitted_data[0, :], dtype=np.float32)
            else:
                init_vals = np.ones((N,), dtype=np.float32) * (self.initial_value if self.initial_value is not None else 0.0)
        else:
            init_vals = np.asarray(initial_value, dtype=np.float32)
            if init_vals.shape == ():
                init_vals = np.full((N,), float(init_vals), dtype=np.float32)
            elif init_vals.shape[0] != N:
                raise ValueError("initial_value must be scalar or length equal to number of channels")
        paths[:, 0, :] = torch.tensor(init_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(num_samples, -1)

        mu = self.mu.to(device) if self.mu is not None else torch.zeros(N, device=device)
        theta = self.theta.to(device) if self.theta is not None else torch.zeros(N, device=device)
        sigma = self.sigma.to(device) if self.sigma is not None else torch.zeros(N, device=device)

        for k in range(L - 1):
            z = torch.randn((num_samples, N), dtype=torch.float32, device=device)
            prev = paths[:, k, :]

            exp_neg_theta = torch.exp(-theta)
            mean = mu * (1 - exp_neg_theta) + prev * exp_neg_theta
            std = sigma * torch.sqrt((1 - exp_neg_theta ** 2) / (2 * theta + 1e-8))
            paths[:, k + 1, :] = mean + std * z

        return paths
