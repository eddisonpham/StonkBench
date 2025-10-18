import torch
import numpy as np
from typing import Optional

from src.models.base.base_model import ParametricModel


class GeometricBrownianMotion(ParametricModel):
    """
    Geometric Brownian Motion parametric model for multichannel time series.

    Assumptions:
      - Input arrays shaped (l, N) where all N channels are feature signals (e.g., OHLC: Open, Close, High, Low).
      - No timestamp channel in input data.
      - This treats each channel as an independent GBM.
      - Time steps are evenly spaced (linear). The model uses dt = 1 as the unit time step.
      - Estimation uses the MLE for constant dt:
            Y_i = log(S_{i+1}) - log(S_i) ~ N((mu - 0.5*sigma^2)*dt, sigma^2 * dt).
    """
    def __init__(
        self,
        length: int,
        num_channels: int,
        initial_price: Optional[float] = 1.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.length = int(length)
        self.num_channels = int(num_channels)
        self.initial_price = initial_price
        self.device = device if device is not None else torch.device("cpu")

        self.mu = None
        self.sigma = None

        self.fitted_data = None

    def fit(self, data):
        """
        Fit GBM parameters to `data` (numpy array or torch tensor) of shape (l, N).

        Estimation is done per-channel using the MLE for constant dt:
            Y_i = log(S_{i+1}) - log(S_i)
            a_hat = mean(Y_i) / dt
            sigma2_hat = var(Y_i) / dt
            mu_hat = a_hat + 0.5 * sigma2_hat
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
            series = data[:, ch].astype(np.float64)
            safe_series = np.clip(series, 1e-12, None)

            if len(safe_series) < 2:
                mu[ch] = 0.0
                sigma[ch] = 0.0
                continue

            logp = np.log(safe_series)
            Y = np.diff(logp)
            num_incr = len(Y)

            if num_incr == 0:
                mu[ch] = 0.0
                sigma[ch] = 0.0
                continue

            # MLE for constant dt: a_hat = mean(Y) / dt
            a_hat = float(np.mean(Y))

            # sigma^2_hat = var(Y) / dt
            sigma2_hat = np.var(Y, ddof=1)
            sigma2_hat = max(sigma2_hat, 0.0)
            sigma_hat = float(np.sqrt(sigma2_hat))

            mu_hat = a_hat + 0.5 * sigma2_hat

            mu[ch] = mu_hat
            sigma[ch] = sigma_hat

        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        
        return {"mu": self.mu, "sigma": self.sigma}

    def generate(self, num_samples: int, initial_price: Optional[np.ndarray] = None,
        output_length: Optional[int] = None, seed: Optional[int] = None
    ):
        """
        Generate `num_samples` independent sample paths of length `output_length` (or self.length if None).

        Returns a torch tensor of shape (num_samples, output_length, num_channels).
        All channels are simulated feature channels.

        OHLC handling assumptions:
          - If the input channels are interpreted as [Open, Close, High, Low] (or subset),
            we simulate Close via GBM, set Open = previous Close, and High/Low = max/min(Open, Close).
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = self.device
        L = int(output_length) if output_length is not None else int(self.length)
        N = self.num_channels

        paths = torch.zeros((num_samples, L, N), dtype=torch.float32, device=device)

        if initial_price is None:
            if hasattr(self, "fitted_data") and self.fitted_data is not None:
                init_vals = np.asarray(self.fitted_data[0, :], dtype=np.float32)
            else:
                init_vals = np.ones((N,), dtype=np.float32) * (self.initial_price if self.initial_price is not None else 1.0)
        else:
            init_vals = np.asarray(initial_price, dtype=np.float32)
            if init_vals.shape == ():
                init_vals = np.full((N,), float(init_vals), dtype=np.float32)
            elif init_vals.shape[0] != N:
                raise ValueError("initial_price must be scalar or length equal to number of channels")

        paths[:, 0, :] = torch.tensor(init_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(num_samples, -1)

        if self.mu is None or self.sigma is None:
            mu = torch.zeros((N,), dtype=torch.float32, device=device)
            sigma = torch.zeros((N,), dtype=torch.float32, device=device)
        else:
            mu = self.mu.to(device)
            sigma = self.sigma.to(device)

        for k in range(L - 1):
            z = torch.randn((num_samples, N), dtype=torch.float32, device=device)
            drift = (mu - 0.5 * sigma * sigma)
            diffusion = sigma * z
            prev = paths[:, k, :]
            factor = torch.exp(drift.unsqueeze(0) + diffusion)
            new_price = prev * factor
            paths[:, k + 1, :] = new_price

        return paths
