import torch
import numpy as np
from typing import Optional
from scipy.optimize import minimize

from src.models.base.base_model import ParametricModel


class GARCH11(ParametricModel):
    """
    GARCH(1,1) parametric model for multichannel financial time series.

    Assumptions:
      - Input arrays shaped (l, N) where channel 0 is timestamp (monotonic),
        channels 1..N-1 are the signals (e.g., OHLC: Open, Close, High, Low).
      - This treats each non-time channel as an independent GARCH(1,1) process.
      - GARCH models the conditional variance of log-returns.
      - Timestamps can be unevenly spaced. The model is discrete-time and treats each
        observed step as one unit of time.
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
        
        # Store initial conditions for simulation
        self._initial_sigma2 = None  # (channels,) tensor
        self._initial_epsilon = None  # (channels,) tensor

        self.fitted_data = None
        self.timestamps = None

    def fit(self, data):
        """
        Fit GARCH(1,1) parameters to `data` (numpy array or torch tensor) of shape (l, N).

        Estimation is done per-channel (channels 1..N) using MLE for GARCH(1,1) on log-returns:
            r_t = log(S_t / S_{t-1})
            r_t = mu + epsilon_t
            epsilon_t = sigma_t * z_t,  z_t ~ N(0,1)
            sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
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
        self.timestamps = data[:, 0].astype(np.float64)

        price_channels = self.num_channels - 1
        mu = np.zeros(price_channels, dtype=np.float64)
        omega = np.zeros(price_channels, dtype=np.float64)
        alpha = np.zeros(price_channels, dtype=np.float64)
        beta = np.zeros(price_channels, dtype=np.float64)
        
        final_sigma2_arr = np.zeros(price_channels, dtype=np.float64)
        final_epsilon_arr = np.zeros(price_channels, dtype=np.float64)

        for ch in range(price_channels):
            series = data[:, ch + 1].astype(np.float64)
            safe_series = np.clip(series, 1e-12, None)

            if len(safe_series) < 2:
                mu[ch] = 0.0
                omega[ch] = 1e-6
                alpha[ch] = 0.05
                beta[ch] = 0.9
                continue

            # Compute log-returns
            r = np.diff(np.log(safe_series)) 
            T_ret = len(r)

            if T_ret == 0:
                mu[ch] = 0.0
                omega[ch] = 1e-6
                alpha[ch] = 0.05
                beta[ch] = 0.9
                continue

            # --- MLE log-likelihood ---
            def neg_loglik(params):
                mu_p, omega_p, alpha_p, beta_p = params
                
                # Constraints: positivity and persistence
                if omega_p <= 1e-8 or alpha_p < 0 or beta_p < 0 or alpha_p + beta_p >= 0.9999:
                    return 1e10
                
                # Residuals (epsilon_t = r_t - mu)
                eps = r - mu_p
                
                # Initialize conditional variance
                sigma2 = np.zeros(T_ret, dtype=np.float64)
                sigma2[0] = np.var(eps) if np.var(eps) > 1e-10 else 1e-6
                
                ll = 0.0
                for t in range(1, T_ret):
                    # GARCH(1,1) recursion
                    sigma2[t] = omega_p + alpha_p * eps[t - 1] ** 2 + beta_p * sigma2[t - 1]
                    
                    # Log-likelihood contribution
                    ll += 0.5 * (np.log(sigma2[t] + 1e-10) + eps[t] ** 2 / (sigma2[t] + 1e-10))
                    
                # Store final values for simulation initialization
                nonlocal final_sigma2_arr, final_epsilon_arr
                final_sigma2_arr[ch] = sigma2[-1]
                final_epsilon_arr[ch] = eps[-1]
                    
                return ll

            # Initial guess
            mu0 = np.mean(r)
            uncon_var = np.var(r) 
            omega0 = 0.1 * uncon_var
            alpha0 = 0.05
            beta0 = 0.9 
            x0 = [mu0, omega0, alpha0, beta0]

            bounds = [(-np.inf, np.inf), (1e-8, np.inf), (0.0, 0.999), (0.0, 0.999)]
            
            res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)
            mu[ch], omega[ch], alpha[ch], beta[ch] = res.x
            
            # Run once more to capture final state
            neg_loglik(res.x)

        # Store parameters as tensors
        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.omega = torch.tensor(omega, dtype=torch.float32, device=self.device)
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        self.beta = torch.tensor(beta, dtype=torch.float32, device=self.device)
        
        self._initial_sigma2 = torch.tensor(final_sigma2_arr, dtype=torch.float32, device=self.device)
        self._initial_epsilon = torch.tensor(final_epsilon_arr, dtype=torch.float32, device=self.device)
        self._fitted_dt = np.diff(self.timestamps)

        return {"mu": self.mu, "omega": self.omega, "alpha": self.alpha, "beta": self.beta}

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

    def generate(self, num_samples: int, initial_value: Optional[np.ndarray] = None,
        output_length: Optional[int] = None, seed: Optional[int] = None,
        linear_timestamps: Optional[bool] = False
    ):
        """
        Generate `num_samples` independent sample paths of length `output_length` (or self.length if None)
        for the GARCH(1,1) process.

        Returns a torch tensor of shape (num_samples, output_length, num_channels).
        Channel 0: timestamps (constructed from fitted timestamps if available, else 0..L-1)
        Channels 1..N-1: simulated price channels.

        GARCH(1,1) assumptions:
        - Log-return dynamics: r_t = mu + epsilon_t, where epsilon_t = sigma_t * z_t
        - Conditional variance: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        - Price evolution: S_t = S_{t-1} * exp(r_t)
        - Multi-channel support
        - Initial values can be provided, else fitted data or default is used
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = self.device
        L = int(output_length) if output_length is not None else int(self.length)
        N = self.num_channels
        num_channels = N - 1

        dt_seq = self._get_dt_sequence(L, linear_timestamps)

        # --- timestamps ---
        t0 = self.timestamps[0] if hasattr(self, "timestamps") and self.timestamps is not None and len(self.timestamps) > 0 else 0.0
        ts = [t0]
        for d in dt_seq:
            ts.append(ts[-1] + float(d))
        timestamps = torch.tensor(np.array(ts, dtype=np.float32)[:L], dtype=torch.float32, device=device)

        paths = torch.zeros((num_samples, L, N), dtype=torch.float32, device=device)
        paths[:, :, 0] = timestamps.unsqueeze(0).expand(num_samples, -1)

        # --- initial values ---
        if initial_value is None:
            if hasattr(self, "fitted_data") and self.fitted_data is not None:
                init_vals = np.asarray(self.fitted_data[0, 1:], dtype=np.float32)
            else:
                init_vals = np.full((num_channels,), float(self.initial_value if self.initial_value is not None else 1.0), dtype=np.float32)
        else:
            init_vals = np.asarray(initial_value, dtype=np.float32)
            if init_vals.shape == ():
                init_vals = np.full((num_channels,), float(init_vals), dtype=np.float32)
            elif init_vals.shape[0] != num_channels:
                raise ValueError("initial_value must be scalar or length equal to number of channels")
        paths[:, 0, 1:] = torch.tensor(init_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(num_samples, -1)

        # --- parameters ---
        mu = self.mu.to(device) if self.mu is not None else torch.zeros(num_channels, device=device)
        omega = self.omega.to(device) if self.omega is not None else torch.full((num_channels,), 1e-6, device=device)
        alpha = self.alpha.to(device) if self.alpha is not None else torch.full((num_channels,), 0.05, device=device)
        beta = self.beta.to(device) if self.beta is not None else torch.full((num_channels,), 0.9, device=device)
        
        # --- initialize volatility and residuals ---
        if self._initial_sigma2 is not None and self._initial_epsilon is not None:
            sigma2_prev = self._initial_sigma2.to(device).unsqueeze(0).expand(num_samples, -1).clone()
            eps_prev = self._initial_epsilon.to(device).unsqueeze(0).expand(num_samples, -1).clone()
        else:
            # Fallback: unconditional variance
            uncon_sigma2 = omega / torch.clamp(1 - alpha - beta, min=1e-6)
            sigma2_prev = uncon_sigma2.unsqueeze(0).expand(num_samples, -1).clone()
            eps_prev = torch.zeros((num_samples, num_channels), dtype=torch.float32, device=device)

        # --- simulate paths ---
        for k in range(L - 1):
            prev = paths[:, k, 1:]

            # GARCH(1,1) conditional variance
            sigma2 = omega + alpha * eps_prev ** 2 + beta * sigma2_prev
            
            # Generate standard normal innovations
            z = torch.randn((num_samples, num_channels), dtype=torch.float32, device=device)
            
            # Compute residuals
            eps = torch.sqrt(torch.clamp(sigma2, min=1e-8)) * z
            
            # Log-return
            log_return = mu + eps
            
            # Update price: S_t = S_{t-1} * exp(r_t)
            paths[:, k + 1, 1:] = prev * torch.exp(log_return)
            
            # Update state for next iteration
            sigma2_prev = sigma2
            eps_prev = eps

        return paths