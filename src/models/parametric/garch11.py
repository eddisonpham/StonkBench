import torch
import numpy as np
from typing import Optional
from scipy.optimize import minimize

from src.models.base.base_model import ParametricModel


class GARCH11(ParametricModel):
    """
    GARCH(1,1) parametric model for multichannel financial time series.

    NOTE ON UNEVEN TIMESTAMPS: This is a discrete-time GARCH(1,1) model. 
    It is applied to the *sequence* of returns, effectively treating each
    observed step as one unit of time, regardless of the real time difference.
    The parameters are thus sequence-dependent, not time-dependent.
    """

    def __init__(self, length: int, num_channels: int, initial_value: Optional[float] = 0.0,
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
        self._initial_sigma2 = None # (channels,) tensor
        self._initial_epsilon = None # (channels,) tensor

        self.fitted_data = None
        self.timestamps = None

    # -------------------
    # --- Estimation ---
    # -------------------
    def fit(self, data):
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

        channels = self.num_channels - 1
        mu = np.zeros(channels, dtype=np.float32)
        omega = np.zeros(channels, dtype=np.float32)
        alpha = np.zeros(channels, dtype=np.float32)
        beta = np.zeros(channels, dtype=np.float32)
        
        final_sigma2_arr = np.zeros(channels, dtype=np.float32)
        final_epsilon_arr = np.zeros(channels, dtype=np.float32)


        for ch in range(channels):
            series = data[:, ch + 1].astype(np.float64)
            if len(series) < 2:
                mu[ch] = series[0] if len(series) > 0 else 0.0
                omega[ch] = alpha[ch] = beta[ch] = 0.0
                continue

            r = np.diff(series) 
            T_ret = len(r)

            # --- MLE log-likelihood ---
            def neg_loglik(params):
                mu_p, omega_p, alpha_p, beta_p = params
                
                # C.R.: Persistence check (alpha + beta < 1) and positivity
                if omega_p <= 1e-8 or alpha_p < 0 or beta_p < 0 or alpha_p + beta_p >= 0.9999:
                    return 1e10
                
                # Residuals (epsilon_t = r_t - mu)
                eps = r - mu_p
                
                # Initialize conditional variance: typically using the unconditional variance or sample variance
                # Using sample variance of residuals for safety, as unconditional variance needs alpha+beta < 1
                sigma2 = np.zeros(T_ret, dtype=np.float64)
                sigma2[0] = np.var(eps) 
                
                ll = 0.0
                for t in range(1, T_ret):
                    # GARCH(1,1) update: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
                    sigma2[t] = omega_p + alpha_p * eps[t - 1] ** 2 + beta_p * sigma2[t - 1]
                    
                    # Log-likelihood contribution (from Normal distribution)
                    # Sum of [ 0.5 * log(sigma2_t) + 0.5 * epsilon_t^2 / sigma2_t ]
                    ll += 0.5 * (np.log(sigma2[t] + 1e-10) + eps[t] ** 2 / sigma2[t])
                    
                # Store final values for simulation initialization
                nonlocal final_sigma2_arr, final_epsilon_arr
                final_sigma2_arr[ch] = sigma2[-1]
                final_epsilon_arr[ch] = eps[-1]
                    
                return ll

            # Initial guess
            mu0 = np.mean(r)
            # Use a slightly more informed guess for unconditional variance
            uncon_var = np.var(r) 
            omega0 = 0.1 * uncon_var
            alpha0 = 0.05
            beta0 = 0.9 
            x0 = [mu0, omega0, alpha0, beta0]

            # Set bounds for L-BFGS-B (omega > 0, alpha, beta >= 0, alpha + beta < 1)
            bounds = [(-np.inf, np.inf), (1e-8, np.inf), (0.0, 0.999), (0.0, 0.999)]
            
            # NOTE: The fit function must be run once more after minimization 
            # to correctly capture the final sigma2 and epsilon
            res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)
            mu[ch], omega[ch], alpha[ch], beta[ch] = res.x.astype(np.float32)
            
            # CRITICAL FIX 1: Run the loglik function one last time with optimal parameters
            # to set final_sigma2_arr and final_epsilon_arr for the generate method.
            neg_loglik(res.x) 


        # store parameters
        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.omega = torch.tensor(omega, dtype=torch.float32, device=self.device)
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        self.beta = torch.tensor(beta, dtype=torch.float32, device=self.device)
        
        self._initial_sigma2 = torch.tensor(final_sigma2_arr, dtype=torch.float32, device=self.device)
        self._initial_epsilon = torch.tensor(final_epsilon_arr, dtype=torch.float32, device=self.device)


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
            linear_timestamps: Optional[bool] = None
        ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = self.device
        L = int(output_length) if output_length is not None else int(self.length)
        N = self.num_channels
        channels = N - 1

        dt_seq = self._get_dt_sequence(L, linear_timestamps=linear_timestamps)

        # ... (Timestamp and initial value setup is correct and unchanged) ...
        # timestamps
        if hasattr(self, "timestamps") and self.timestamps is not None and len(self.timestamps) == L:
            timestamps = torch.tensor(self.timestamps, dtype=torch.float32, device=device)
        else:
            ts = [0.0]
            for d in dt_seq:
                ts.append(ts[-1] + float(d))
            timestamps = torch.tensor(ts[:L], dtype=torch.float32, device=device)

        paths = torch.zeros((num_samples, L, N), dtype=torch.float32, device=device)
        paths[:, :, 0] = timestamps.unsqueeze(0).expand(num_samples, -1)

        # initial values
        if initial_value is None:
            if hasattr(self, "fitted_data") and self.fitted_data is not None:
                init_vals = np.asarray(self.fitted_data[0, 1:], dtype=np.float32)
            else:
                init_vals = np.full((channels,), float(self.initial_value), dtype=np.float32)
        else:
            init_vals = np.asarray(initial_value, dtype=np.float32)
            if init_vals.shape == ():
                init_vals = np.full((channels,), float(init_vals), dtype=np.float32)
            elif init_vals.shape[0] != channels:
                raise ValueError("initial_value must be scalar or length equal to number of channels")
        paths[:, 0, 1:] = torch.tensor(init_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(num_samples, -1)
        
        # parameters (including safe defaults if not fitted)
        mu = self.mu.to(device) if self.mu is not None else torch.zeros(channels, device=device)
        omega = self.omega.to(device) if self.omega is not None else torch.full((channels,), 0.01, device=device)
        alpha = self.alpha.to(device) if self.alpha is not None else torch.zeros(channels, device=device)
        beta = self.beta.to(device) if self.beta is not None else torch.zeros(channels, device=device)
        
        # --- CRITICAL FIX 2 & 3: Initialization for the first step (k=1) ---
        if self._initial_sigma2 is not None and self._initial_epsilon is not None:
            # If fitted, use the last calculated sigma2 and epsilon from the fitted series
            sigma2_prev = self._initial_sigma2.unsqueeze(0).expand(num_samples, -1).clone()
            eps_prev = self._initial_epsilon.unsqueeze(0).expand(num_samples, -1).clone()
        else:
            # Fallback for unfitted model: use unconditional variance and zero residual
            uncon_sigma2 = omega / (1 - alpha - beta)
            sigma2_prev = uncon_sigma2.unsqueeze(0).expand(num_samples, -1).clone()
            eps_prev = torch.zeros((num_samples, channels), dtype=torch.float32, device=device)

        # simulate
        for k in range(1, L):
            # 1. Calculate Conditional Variance (sigma_t^2)
            # sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
            sigma2 = omega + alpha * eps_prev ** 2 + beta * sigma2_prev
            
            # 2. Calculate New Residual (epsilon_t = sigma_t * z_t)
            z = torch.randn((num_samples, channels), dtype=torch.float32, device=device)
            eps = torch.sqrt(torch.clamp(sigma2, min=1e-8)) * z # Clamp to prevent negative variance
            
            # 3. Calculate New Price (S_t = S_{t-1} + r_t = S_{t-1} + mu + epsilon_t)
            paths[:, k, 1:] = paths[:, k - 1, 1:] + mu + eps
            
            # 4. Update Previous States for the next step (k+1)
            sigma2_prev = sigma2
            eps_prev = eps 

        return paths