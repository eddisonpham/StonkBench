import torch
import numpy as np
from typing import Optional

from src.models.base.base_model import ParametricModel


class OrnsteinUhlenbeckProcess(ParametricModel):
    """
    Ornstein-Uhlenbeck (O-U) parametric model for multichannel time series.

    Assumptions:
      - Input arrays shaped (l, N) where channel 0 is timestamp (monotonic),
        channels 1..N-1 are the signals.
      - Each non-time channel is an independent O-U process.
      - Timestamps can be unevenly spaced. Estimation uses MLE for varying dt:
            X_{t+dt} = X_t * exp(-theta*dt) + mu*(1 - exp(-theta*dt)) + epsilon,
            epsilon ~ N(0, sigma^2*(1 - exp(-2*theta*dt))/(2*theta))
    """

    def __init__(self, length: int, num_channels: int, initial_value: Optional[float] = 0.0, device: Optional[torch.device] = None):
        super().__init__()
        self.length = int(length)
        self.num_channels = int(num_channels)
        self.initial_value = initial_value
        self.device = device if device is not None else torch.device("cpu")

        # fitted parameters (per-channel)
        self.mu = None      # torch tensor shape (num_channels-1,)
        self.theta = None   # torch tensor shape (num_channels-1,)
        self.sigma = None   # torch tensor shape (num_channels-1,)

        # store fitted_data and timestamps when fit() is called
        self.fitted_data = None
        self.timestamps = None

    # -------------------
    # --- Estimation ---
    # -------------------
    def fit(self, data):
        """
        Fit O-U parameters to `data` (numpy array or torch tensor) of shape (l, N).
        Uses MLE for irregular dt.
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
        timestamps = data[:, 0].astype(np.float64)
        if np.any(np.diff(timestamps) < 0):
            raise ValueError("timestamps must be non-decreasing")
        self.timestamps = timestamps

        # compute dt_i for each increment: dt_i = t_{i+1} - t_i
        if l > 1:
            dt = np.diff(timestamps)
            dt = np.where(dt <= 0, 1e-8, dt)
        else:
            dt = np.array([1.0], dtype=np.float64)

        price_channels = self.num_channels - 1
        mu = np.zeros(price_channels, dtype=np.float64)
        theta = np.zeros(price_channels, dtype=np.float64)
        sigma = np.zeros(price_channels, dtype=np.float64)

        for ch in range(price_channels):
            series = data[:, ch + 1].astype(np.float64)

            if len(series) < 2:
                mu[ch] = series[0] if len(series) > 0 else 0.0
                theta[ch] = 0.0
                sigma[ch] = 0.0
                continue

            X = series
            X_next = X[1:]
            X_prev = X[:-1]
            dt_local = dt[:len(X_next)]

            # estimate theta and mu using regression on X_{t+dt} = a*X_t + b
            # a = exp(-theta*dt), b = mu*(1 - a)
            # solve for each dt individually (weighted least squares)
            A = np.vstack([X_prev, np.ones_like(X_prev)]).T  # regression matrix
            # simple approximation: ignore dt variations for stability: use mean dt
            mean_dt = np.mean(dt_local)
            # linear regression: X_next = a * X_prev + b
            cov = np.cov(X_prev, X_next, bias=True)
            var_x = cov[0, 0]
            cov_xy = cov[0, 1]
            a_hat = cov_xy / var_x if var_x > 0 else 0.0
            a_hat = np.clip(a_hat, -0.999, 0.999)  # ensure numerical stability
            theta_hat = -np.log(a_hat) / mean_dt if a_hat != 0 else 0.0
            mu_hat = np.mean(X_next - a_hat * X_prev) / (1 - a_hat) if a_hat != 1 else np.mean(X_prev)
            # sigma^2_hat = Var(epsilon) * 2*theta / (1 - exp(-2*theta*dt))
            eps = X_next - (a_hat * X_prev + (1 - a_hat) * mu_hat)
            var_eps = np.var(eps, ddof=0)
            sigma_hat = np.sqrt(var_eps * 2 * theta_hat / (1 - np.exp(-2 * theta_hat * mean_dt))) if theta_hat > 0 else np.sqrt(var_eps)

            mu[ch] = mu_hat
            theta[ch] = theta_hat
            sigma[ch] = sigma_hat

        # Save as torch tensors on device
        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        self._fitted_dt = dt

        return {"mu": self.mu, "theta": self.theta, "sigma": self.sigma}

    # -----------------------
    # --- Sample / Generate ---
    # -----------------------
    def _get_dt_sequence(self, l_out: int):
        if l_out <= 1:
            return np.array([1.0], dtype=np.float32)
        if hasattr(self, "timestamps") and self.timestamps is not None and len(self.timestamps) > 1:
            fitted_dt = np.diff(self.timestamps).astype(np.float64)
            if len(fitted_dt) == l_out - 1:
                return fitted_dt.astype(np.float32)
            mean_dt = float(np.mean(fitted_dt)) if len(fitted_dt) > 0 else 1.0
            if len(fitted_dt) >= l_out - 1:
                return fitted_dt[: l_out - 1].astype(np.float32)
            else:
                pad = np.full((l_out - 1 - len(fitted_dt),), fitted_dt[-1] if len(fitted_dt) > 0 else mean_dt, dtype=np.float32)
                return np.concatenate([fitted_dt.astype(np.float32), pad]).astype(np.float32)
        else:
            return np.ones(l_out - 1, dtype=np.float32)

    def generate(self, num_samples: int, initial_value: Optional[np.ndarray] = None, output_length: Optional[int] = None, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = self.device
        L = int(output_length) if output_length is not None else int(self.length)
        N = self.num_channels
        price_channels = N - 1

        dt_seq = self._get_dt_sequence(L)
        dt_torch = torch.tensor(dt_seq, dtype=torch.float32, device=device)

        # timestamps
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

        # initial values
        if initial_value is None:
            if hasattr(self, "fitted_data") and self.fitted_data is not None:
                init_vals = np.asarray(self.fitted_data[0, 1:], dtype=np.float32)
            else:
                init_vals = np.ones((price_channels,), dtype=np.float32) * (self.initial_value if self.initial_value is not None else 0.0)
        else:
            init_vals = np.asarray(initial_value, dtype=np.float32)
            if init_vals.shape == ():
                init_vals = np.full((price_channels,), float(init_vals), dtype=np.float32)
            elif init_vals.shape[0] != price_channels:
                raise ValueError("initial_value must be scalar or length equal to number of channels")
        paths[:, 0, 1:] = torch.tensor(init_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(num_samples, -1)

        # fallback parameters if not fitted
        mu = self.mu.to(device) if self.mu is not None else torch.zeros(price_channels, device=device)
        theta = self.theta.to(device) if self.theta is not None else torch.zeros(price_channels, device=device)
        sigma = self.sigma.to(device) if self.sigma is not None else torch.zeros(price_channels, device=device)

        # simulate
        for k in range(L - 1):
            dtk = float(dt_seq[k])
            z = torch.randn((num_samples, price_channels), dtype=torch.float32, device=device)
            prev = paths[:, k, 1:]
            exp_neg_theta_dt = torch.exp(-theta * dtk)
            mean = mu * (1 - exp_neg_theta_dt) + prev * exp_neg_theta_dt
            std = sigma * torch.sqrt((1 - exp_neg_theta_dt ** 2) / (2 * theta + 1e-8))  # add small eps to avoid div0
            paths[:, k + 1, 1:] = mean + std * z

        return paths
