import torch
import numpy as np
from typing import Optional

from src.models.base.base_model import ParametricModel


class GeometricBrownianMotion(ParametricModel):
    """
    Geometric Brownian Motion parametric model for multichannel time series.

    Assumptions:
      - Input arrays shaped (l, N) where channel 0 is timestamp (monotonic),
        channels 1..N-1 are the signals (e.g., OHLCV: Open, Close, High, Low, Volume).
      - This treats each non-time channel as an independent GBM.
      - Timestamps can be unevenly spaced. Estimation uses the MLE that handles varying dt:
            Y_i = log(S_{t_{i+1}}) - log(S_{t_i}) ~ N((mu - 0.5*sigma^2)*dt_i, sigma^2 * dt_i).
    """
    def __init__(self, length: int, num_channels: int, initial_price: Optional[float] = 1.0, device: Optional[torch.device] = None):
        super().__init__()
        self.length = int(length)
        self.num_channels = int(num_channels)
        self.initial_price = initial_price
        self.device = device if device is not None else torch.device("cpu")

        self.mu = None
        self.sigma = None

        self.fitted_data = None
        self.timestamps = None

    def fit(self, data):
        """
        Fit GBM parameters to `data` (numpy array or torch tensor) of shape (l, N).

        Estimation is done per-channel (channels 1..N) using the MLE for uneven dt:
            Y_i = log(S_{t_{i+1}}) - log(S_{t_i})
            a_hat = sum(Y_i) / sum(dt_i)
            sigma2_hat = (1/N) * sum( (Y_i - a_hat*dt_i)^2 / dt_i )
            mu_hat = a_hat + 0.5 * sigma2_hat
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2D array with shape (l, N)")

        self.fitted_data = data.copy()
        self.timestamps = data[:, 0].astype(np.float64)

        dt = np.diff(self.timestamps)

        price_channels = self.num_channels - 1
        mu = np.zeros(price_channels, dtype=np.float64)
        sigma = np.zeros(price_channels, dtype=np.float64)

        for ch in range(price_channels):
            series = data[:, ch + 1].astype(np.float64)
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

            # MLE: a_hat = sum(Y) / sum(dt)
            sum_dt = float(np.sum(dt))
            a_hat = float(np.sum(Y)) / sum_dt if sum_dt > 0 else 0.0

            # sigma^2_hat = (1/N) * sum( (Y - a_hat*dt)^2 / dt )
            resid = Y - a_hat * dt
            sigma2_hat = np.sum((resid ** 2) / dt) / num_incr
            sigma2_hat = max(sigma2_hat, 0.0)
            sigma_hat = float(np.sqrt(sigma2_hat))

            mu_hat = a_hat + 0.5 * sigma2_hat

            mu[ch] = mu_hat
            sigma[ch] = sigma_hat

        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        self._fitted_dt = dt
        return {"mu": self.mu, "sigma": self.sigma}

    def _get_dt_sequence(self, l_out: int, linear_timestamps: bool = False):
        """
        Return a dt sequence of length l_out (dt for steps between samples).

        If linear_timestamps=True, generate l_out timestamps that are linearly spaced
        with each increment equal to mean(dt). The output dt_seq is of length l_out-1.

        If linear_timestamps=False, use fitted per-step dt where possible, and fill by mean_dt or last dt.

        Args:
            l_out (int): Number of output timestamps (so returned sequence is length l_out-1)
            linear_timestamps (bool): If True, ignore fitted timestamps and generate uniform dt sequence.

        Returns:
            numpy.ndarray: Array of dt increments of length l_out-1.
        """
        assert hasattr(self, "timestamps"), "Timestamps attribute is required to generate a dt sequence, but was not found on this object."
        
        fitted_dt = np.diff(self.timestamps).astype(np.float64)
        mean_dt = float(np.mean(fitted_dt)) if len(fitted_dt) > 0 else 1.0

        if linear_timestamps:
            return np.full((l_out - 1,), mean_dt, dtype=np.float32)
        else:
            if len(fitted_dt) == l_out - 1:
                return fitted_dt.astype(np.float32)
            if len(fitted_dt) > l_out - 1:
                return fitted_dt[: l_out - 1].astype(np.float32)
            else:
                pad = np.full((l_out - 1 - len(fitted_dt),), fitted_dt[-1] if len(fitted_dt) > 0 else mean_dt, dtype=np.float32)
                return np.concatenate([fitted_dt.astype(np.float32), pad]).astype(np.float32)

    def generate(self, num_samples: int, initial_price: Optional[np.ndarray] = None,
        output_length: Optional[int] = None, seed: Optional[int] = None,
        linear_timestamps: Optional[bool] = False
    ):
        """
        Generate `num_samples` independent sample paths of length `output_length` (or self.length if None).

        Returns a torch tensor of shape (num_samples, output_length, num_channels).
        Channel 0: timestamps (constructed from fitted timestamps if available, else 0..L-1)
        Channels 1..N-1: simulated channels.

        OHLC handling assumptions:
          - If the input channels are interpreted as [Open, Close, High, Low] (or subset),
            we simulate Close via GBM, set Open = previous Close, and High/Low = max/min(Open, Close).
            Volume is simulated as GBM as well (or you can override).
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = self.device
        L = int(output_length) if output_length is not None else int(self.length)
        N = self.num_channels
        price_channels = N - 1

        dt_seq = self._get_dt_sequence(L, linear_timestamps=linear_timestamps)
        dt_torch = torch.tensor(dt_seq, dtype=torch.float32, device=device)

        if hasattr(self, "timestamps") and self.timestamps is not None and len(self.timestamps) == L:
            timestamps = torch.tensor(self.timestamps, dtype=torch.float32, device=device)
        else:
            t0 = 0.0
            ts = [t0]
            for d in dt_seq:
                ts.append(ts[-1] + float(d))
            timestamps = torch.tensor(np.array(ts, dtype=np.float32)[:L], dtype=torch.float32, device=device)

        paths = torch.zeros((num_samples, L, N), dtype=torch.float32, device=device)
        paths[:, :, 0] = timestamps.unsqueeze(0).expand(num_samples, -1)

        if initial_price is None:
            if hasattr(self, "fitted_data") and self.fitted_data is not None:
                init_vals = np.asarray(self.fitted_data[0, 1:], dtype=np.float32)
            else:
                init_vals = np.ones((price_channels,), dtype=np.float32) * (self.initial_price if self.initial_price is not None else 1.0)
        else:
            init_vals = np.asarray(initial_price, dtype=np.float32)
            if init_vals.shape == ():
                init_vals = np.full((price_channels,), float(init_vals), dtype=np.float32)
            elif init_vals.shape[0] != price_channels:
                raise ValueError("initial_price must be scalar or length equal to number of price channels")

        paths[:, 0, 1:] = torch.tensor(init_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(num_samples, -1)

        if self.mu is None or self.sigma is None:
            mu = torch.zeros((price_channels,), dtype=torch.float32, device=device)
            sigma = torch.zeros((price_channels,), dtype=torch.float32, device=device)
        else:
            mu = self.mu.to(device)
            sigma = self.sigma.to(device)

        for k in range(L - 1):
            dtk = float(dt_seq[k])
            z = torch.randn((num_samples, price_channels), dtype=torch.float32, device=device)
            drift = (mu - 0.5 * sigma * sigma) * dtk
            diffusion = sigma * (np.sqrt(dtk)) * z
            prev = paths[:, k, 1:]
            factor = torch.exp(drift.unsqueeze(0) + diffusion)
            new_price = prev * factor
            paths[:, k + 1, 1:] = new_price

        return paths

