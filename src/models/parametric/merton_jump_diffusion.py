import torch
import numpy as np
from typing import Optional

from src.models.base.base_model import ParametricModel


class MertonJumpDiffusion(ParametricModel):
    """
    Merton Jump Diffusion (MJD) parametric model for multichannel time series.

    Assumptions:
      - Input arrays shaped (l, N) where all N channels are feature signals.
      - No timestamp channel in input data.
      - SDE for price S_t: dS_t = S_{t^-} ( (mu - lambda*E[e^Y - 1]) dt + sigma dW_t + (e^Y - 1) dN_t )
      - The drift is adjusted by the jump component's expectation to keep mu as the 
        expected log-return (log(S_t/S_0) / t).
      - Time steps are evenly spaced (linear) with unit dt.
    """
    def __init__(self, length: int, num_channels: int, initial_value: Optional[float] = 1.0, 
                 device: Optional[torch.device] = None):
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

    def fit(self, data):
        """
        Fit the Merton Jump Diffusion model parameters to `data`.

        Estimates per-channel parameters via a jump-separation heuristic:
          - mu: Continuous drift (with jump expectation correction).
          - sigma: Diffusive volatility (excluding jumps).
          - lamb: Estimated Poisson jump intensity (frequency per unit time).
          - mu_j: Mean jump size (log-return), for jump events.
          - sigma_j: Volatility of jump size.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2D array with shape (l, N)")

        self.fitted_data = data.copy()

        mu = np.zeros(self.num_channels, dtype=np.float64)
        sigma = np.zeros(self.num_channels, dtype=np.float64)
        lamb = np.zeros(self.num_channels, dtype=np.float64)
        mu_j = np.zeros(self.num_channels, dtype=np.float64)
        sigma_j = np.zeros(self.num_channels, dtype=np.float64)

        for ch in range(self.num_channels):
            series = data[:, ch].astype(np.float64)
            returns = np.diff(np.log(series)) 

            # Total mean and std
            temp_mu_total = np.mean(returns)
            temp_sigma_total = np.std(returns, ddof=0)

            # Jump detection via threshold
            C = 4.0
            jump_threshold = C * temp_sigma_total
            jumps = np.abs(returns - temp_mu_total) > jump_threshold
            jump_vals = returns[jumps]

            # Estimate jump parameters
            if len(jump_vals) > 1:
                num_jumps = len(jump_vals)
                total_time = len(returns)
                lamb[ch] = num_jumps / total_time if total_time > 0 else 1e-6
                
                mu_j[ch] = np.mean(jump_vals)
                sigma_j[ch] = np.std(jump_vals, ddof=0) + 1e-6
            else:
                lamb[ch] = 1e-6
                mu_j[ch] = 0.0
                sigma_j[ch] = 1e-6

            # Continuous (diffusive) returns
            returns_no_jumps = returns[~jumps]
            if len(returns_no_jumps) > 1:
                mu_diff = np.mean(returns_no_jumps)
                sigma_diff = np.std(returns_no_jumps, ddof=0)

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

        return {"mu": self.mu, "sigma": self.sigma, "lamb": self.lamb, "mu_j": self.mu_j, "sigma_j": self.sigma_j}

    def generate(self, num_samples: int, initial_value: Optional[np.ndarray] = None,
        output_length: Optional[int] = None, seed: Optional[int] = None
    ):
        """
        Generate `num_samples` independent sample paths of length `output_length` (or self.length if None)
        for the Merton Jump Diffusion process.

        Returns a torch tensor of shape (num_samples, output_length, num_channels).
        All channels are simulated feature channels.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = self.device
        L = int(output_length) if output_length is not None else int(self.length)
        N = self.num_channels

        paths = torch.zeros((num_samples, L, N), dtype=torch.float32, device=device)

        # Initial values
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

        # Model parameters
        mu = self.mu.to(device) if self.mu is not None else torch.zeros(N, device=device)
        sigma = self.sigma.to(device) if self.sigma is not None else torch.ones(N, device=device) * 1e-3
        lamb = self.lamb.to(device) if self.lamb is not None else torch.ones(N, device=device) * 1e-6
        mu_j = self.mu_j.to(device) if self.mu_j is not None else torch.zeros(N, device=device)
        sigma_j = self.sigma_j.to(device) if self.sigma_j is not None else torch.ones(N, device=device) * 1e-3

        mean_jump_factor = torch.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        jump_drift_correction = lamb * mean_jump_factor

        # Generate paths
        for k in range(L - 1):
            prev = paths[:, k, :]

            # Continuous diffusion
            log_drift = (mu - 0.5 * sigma**2 - jump_drift_correction)
            z_diff = torch.randn((num_samples, N), dtype=torch.float32, device=device)
            log_continuous = log_drift + sigma * z_diff

            # Jump component
            poisson = torch.poisson(lamb)
            total_jump_std = torch.sqrt(torch.clamp(poisson, min=0.0)) * sigma_j
            z_jumps = torch.randn((num_samples, N), dtype=torch.float32, device=device)
            total_log_jump = poisson * mu_j + total_jump_std * z_jumps

            paths[:, k + 1, :] = prev * torch.exp(log_continuous + total_log_jump)

        return paths
