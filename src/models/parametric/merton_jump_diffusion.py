import torch
import numpy as np
from typing import Optional

from src.models.base.base_model import ParametricModel


class MertonJumpDiffusion(ParametricModel):
    """
    Merton Jump Diffusion (MJD) parametric model for multichannel time series.

    Assumptions:
      - SDE for price S_t: dS_t = S_{t^-} ( (mu - lambda*E[e^Y - 1]) dt + sigma dW_t + (e^Y - 1) dN_t )
      - The drift is adjusted by the jump component's expectation to keep mu as the 
        expected log-return (log(S_t/S_0) / t).
    """
    def __init__(self, length: int, num_channels: int, initial_value: Optional[float] = 1.0, device: Optional[torch.device] = None):
        super().__init__()
        self.length = int(length)
        self.num_channels = int(num_channels)
        self.initial_value = initial_value
        self.device = device if device is not None else torch.device("cpu")

        self.mu = None
        self.sigma = None
        self.lamb = None
        self.mu_j = None
        self.sigma_j = None

        self.fitted_data = None
        self.timestamps = None

    def fit(self, data):
        """
        Fit the Merton Jump Diffusion model parameters to `data`.

        Estimates per-channel parameters via a jump-separation heuristic:
          - mu: Continuous drift (with jump expectation correction).
          - sigma: Diffusive volatility (excluding jumps).
          - lamb: Estimated Poisson jump intensity (frequency).
          - mu_j: Mean jump size (log-return), for jump events.
          - sigma_j: Volatility of jump size.
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
        lamb = np.zeros(price_channels, dtype=np.float64)
        mu_j = np.zeros(price_channels, dtype=np.float64)
        sigma_j = np.zeros(price_channels, dtype=np.float64)

        for ch in range(price_channels):
            series = data[:, ch + 1].astype(np.float64)
            returns = np.diff(np.log(series)) 

            temp_mu_total = np.mean(returns) / dt
            temp_sigma_total = np.std(returns, ddof=0) / np.sqrt(dt)

            C = 4.0
            jump_threshold = C * temp_sigma_total * np.sqrt(dt)
            jumps = np.abs(returns - temp_mu_total * dt) > jump_threshold
            jump_vals = returns[jumps]

            if len(jump_vals) > 1:
                mu_j[ch] = np.mean(jump_vals)
                sigma_j[ch] = np.std(jump_vals, ddof=0) + 1e-6 
            else:
                lamb[ch] = 1e-6 
                mu_j[ch] = 0.0
                sigma_j[ch] = 1e-6

            returns_no_jumps = returns[~jumps]
            dt_no_jumps = dt[~jumps]
            
            if len(returns_no_jumps) > 1 and np.mean(dt_no_jumps) > 1e-9:
                dt_no_jumps = np.mean(dt_no_jumps) 
                
                mu_diff = np.mean(returns_no_jumps) / dt_no_jumps
                sigma_diff = np.std(returns_no_jumps, ddof=0) / np.sqrt(dt_no_jumps)
                
                jump_comp_exp = np.exp(mu_j[ch] + 0.5 * sigma_j[ch]**2) - 1.0
                
                mu[ch] = mu_diff + 0.5 * sigma_diff**2 + lamb[ch] * jump_comp_exp
                sigma[ch] = sigma_diff
            else:
                mu[ch] = temp_mu_total + 0.5 * temp_sigma_total**2 
                sigma[ch] = temp_sigma_total

        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        self.lamb = torch.tensor(lamb, dtype=torch.float32, device=self.device)
        self.mu_j = torch.tensor(mu_j, dtype=torch.float32, device=self.device)
        self.sigma_j = torch.tensor(sigma_j, dtype=torch.float32, device=self.device)
        self._fitted_dt = dt

        return {"mu": self.mu, "sigma": self.sigma, "lamb": self.lamb, "mu_j": self.mu_j, "sigma_j": self.sigma_j}

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
        for the Merton Jump Diffusion process.

        Returns a torch tensor of shape (num_samples, output_length, num_channels).
        Channel 0: timestamps (constructed from fitted timestamps if available, else 0..L-1)
        Channels 1..N-1: simulated channels.

        Merton Jump Diffusion assumptions:
        - Log-price SDE: dS_t / S_t = (mu - 0.5 sigma^2 - lambda E[e^Y-1]) dt + sigma dW_t + dJ_t
        - Jump arrival: Poisson(lambda), jump size Y ~ N(mu_j, sigma_j^2)
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

        mu = self.mu.to(device) if self.mu is not None else torch.zeros(num_channels, device=device)
        sigma = self.sigma.to(device) if self.sigma is not None else torch.ones(num_channels, device=device) * 1e-3
        lamb = self.lamb.to(device) if self.lamb is not None else torch.ones(num_channels, device=device) * 1e-6
        mu_j = self.mu_j.to(device) if self.mu_j is not None else torch.zeros(num_channels, device=device)
        sigma_j = self.sigma_j.to(device) if self.sigma_j is not None else torch.ones(num_channels, device=device) * 1e-3

        mean_jump_factor = torch.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        jump_drift_correction = lamb * mean_jump_factor

        for k in range(L - 1):
            dtk = float(dt_seq[k])
            dtk_t = torch.tensor(dtk, dtype=torch.float32, device=device)
            prev = paths[:, k, 1:]

            log_drift = (mu - 0.5 * sigma**2 - jump_drift_correction) * dtk_t
            z_diff = torch.randn((num_samples, num_channels), dtype=torch.float32, device=device)
            diffusion_increment = sigma * torch.sqrt(dtk_t) * z_diff
            log_continuous = log_drift + diffusion_increment

            poisson = torch.poisson(lamb * dtk_t).float()
            total_jump_std = torch.sqrt(torch.clamp(poisson, min=0.0)) * sigma_j
            z_jumps = torch.randn((num_samples, num_channels), dtype=torch.float32, device=device)
            total_log_jump = poisson * mu_j + total_jump_std * z_jumps

            paths[:, k + 1, 1:] = prev * torch.exp(log_continuous + total_log_jump)

        return paths
