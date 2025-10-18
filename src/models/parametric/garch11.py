import torch
import numpy as np
from typing import Optional
from scipy.optimize import minimize

from src.models.base.base_model import ParametricModel


class GARCH11(ParametricModel):
    """
    GARCH(1,1) parametric model for multichannel financial time series.

    Assumptions:
      - Input arrays shaped (l, N) where all N channels are feature signals (e.g., OHLC: Open, Close, High, Low).
      - No timestamp channel in input data.
      - Each channel is treated as an independent GARCH(1,1) process.
      - GARCH models the conditional variance of log-returns.
      - Time steps are evenly spaced (linear); each step is one unit of discrete time.
    """

    def __init__(self, length: int, num_channels: int, initial_value: Optional[float] = 1.0,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.length = int(length)
        self.num_channels = int(num_channels)
        self.initial_value = initial_value
        self.device = device if device is not None else torch.device("cpu")

        self.mu = None
        self.omega = None
        self.alpha = None
        self.beta = None
        
        self._initial_sigma2 = None  # (channels,) tensor
        self._initial_epsilon = None  # (channels,) tensor

        self.fitted_data = None

    def fit(self, data):
        """
        Fit GARCH(1,1) parameters to `data` (numpy array or torch tensor) of shape (l, N).
        Uses MLE per channel on log-returns.
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

        mu = np.zeros(N, dtype=np.float64)
        omega = np.zeros(N, dtype=np.float64)
        alpha = np.zeros(N, dtype=np.float64)
        beta = np.zeros(N, dtype=np.float64)
        
        final_sigma2_arr = np.zeros(N, dtype=np.float64)
        final_epsilon_arr = np.zeros(N, dtype=np.float64)

        for ch in range(N):
            series = data[:, ch].astype(np.float64)
            safe_series = np.clip(series, 1e-12, None)

            if len(safe_series) < 2:
                mu[ch] = 0.0
                omega[ch] = 1e-6
                alpha[ch] = 0.05
                beta[ch] = 0.9
                continue

            r = np.diff(np.log(safe_series))  # log-returns
            T_ret = len(r)
            if T_ret == 0:
                mu[ch] = 0.0
                omega[ch] = 1e-6
                alpha[ch] = 0.05
                beta[ch] = 0.9
                continue

            def neg_loglik(params):
                mu_p, omega_p, alpha_p, beta_p = params
                if omega_p <= 1e-8 or alpha_p < 0 or beta_p < 0 or alpha_p + beta_p >= 0.9999:
                    return 1e10

                eps = r - mu_p
                sigma2 = np.zeros(T_ret, dtype=np.float64)
                sigma2[0] = np.var(eps) if np.var(eps) > 1e-10 else 1e-6
                ll = 0.0
                for t in range(1, T_ret):
                    sigma2[t] = omega_p + alpha_p * eps[t-1]**2 + beta_p * sigma2[t-1]
                    ll += 0.5 * (np.log(sigma2[t] + 1e-10) + eps[t]**2 / (sigma2[t] + 1e-10))
                
                nonlocal final_sigma2_arr, final_epsilon_arr
                final_sigma2_arr[ch] = sigma2[-1]
                final_epsilon_arr[ch] = eps[-1]
                return ll

            mu0 = np.mean(r)
            uncon_var = np.var(r)
            x0 = [mu0, 0.1*uncon_var, 0.05, 0.9]
            bounds = [(-np.inf, np.inf), (1e-8, np.inf), (0.0, 0.999), (0.0, 0.999)]

            res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)
            mu[ch], omega[ch], alpha[ch], beta[ch] = res.x
            neg_loglik(res.x)  # store final state

        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.omega = torch.tensor(omega, dtype=torch.float32, device=self.device)
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        self.beta = torch.tensor(beta, dtype=torch.float32, device=self.device)

        self._initial_sigma2 = torch.tensor(final_sigma2_arr, dtype=torch.float32, device=self.device)
        self._initial_epsilon = torch.tensor(final_epsilon_arr, dtype=torch.float32, device=self.device)

        return {"mu": self.mu, "omega": self.omega, "alpha": self.alpha, "beta": self.beta}

    def generate(self, num_samples: int, initial_value: Optional[np.ndarray] = None,
                 output_length: Optional[int] = None, seed: Optional[int] = None):
        """
        Generate sample paths for the GARCH(1,1) process.
        Returns tensor of shape (num_samples, output_length, num_channels).
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
                init_vals = np.full((N,), float(self.initial_value if self.initial_value is not None else 1.0), dtype=np.float32)
        else:
            init_vals = np.asarray(initial_value, dtype=np.float32)
            if init_vals.shape == ():
                init_vals = np.full((N,), float(init_vals), dtype=np.float32)
            elif init_vals.shape[0] != N:
                raise ValueError("initial_value must be scalar or length equal to number of channels")
        paths[:, 0, :] = torch.tensor(init_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(num_samples, -1)

        mu = self.mu.to(device) if self.mu is not None else torch.zeros(N, device=device)
        omega = self.omega.to(device) if self.omega is not None else torch.full((N,), 1e-6, device=device)
        alpha = self.alpha.to(device) if self.alpha is not None else torch.full((N,), 0.05, device=device)
        beta = self.beta.to(device) if self.beta is not None else torch.full((N,), 0.9, device=device)

        if self._initial_sigma2 is not None and self._initial_epsilon is not None:
            sigma2_prev = self._initial_sigma2.to(device).unsqueeze(0).expand(num_samples, -1).clone()
            eps_prev = self._initial_epsilon.to(device).unsqueeze(0).expand(num_samples, -1).clone()
        else:
            uncon_sigma2 = omega / torch.clamp(1 - alpha - beta, min=1e-6)
            sigma2_prev = uncon_sigma2.unsqueeze(0).expand(num_samples, -1).clone()
            eps_prev = torch.zeros((num_samples, N), dtype=torch.float32, device=device)

        for k in range(L - 1):
            prev = paths[:, k, :]
            sigma2 = omega + alpha * eps_prev ** 2 + beta * sigma2_prev
            z = torch.randn((num_samples, N), dtype=torch.float32, device=device)
            eps = torch.sqrt(torch.clamp(sigma2, min=1e-8)) * z
            log_return = mu + eps
            paths[:, k + 1, :] = prev * torch.exp(log_return)
            sigma2_prev = sigma2
            eps_prev = eps

        return paths
