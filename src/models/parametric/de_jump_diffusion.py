import torch
import numpy as np
from typing import Optional

from src.models.base.base_model import ParametricModel


class DoubleExponentialJumpDiffusion(ParametricModel):
    """
    Double Exponential Jump Diffusion (DEJD) parametric model for multichannel time series.

    Assumptions:
      - Input arrays shaped (l, N) where all N channels are feature signals.
      - No timestamp channel in input data.
      - SDE for price S_t: dS_t = S_{t^-} ( (mu - lambda * k) dt + sigma dW_t + (J - 1) dN_t )
      - Jumps are double exponential: positive jumps ~ Exp(eta1), negative jumps ~ Exp(eta2)
      - Jump probability p: P(positive jump)
      - Drift is adjusted by the jump expectation k = E[J-1] to keep mu as the expected log-return.
      - Time steps are evenly spaced (linear).
    """
    def __init__(self, length: int, num_channels: int, initial_value: Optional[float] = 1.0,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.length = int(length)
        self.num_channels = int(num_channels)
        self.initial_value = initial_value
        self.device = device if device is not None else torch.device("cpu")

        # Fitted parameters
        self.mu = None        # total log-return drift
        self.sigma = None     # diffusion volatility
        self.lamb = None      # jump intensity
        self.p = None         # probability of positive jump
        self.eta1 = None      # rate of positive jump exponential
        self.eta2 = None      # rate of negative jump exponential

        # Store fitted data
        self.fitted_data = None

    def fit(self, data):
        """
        Fit approximate parameters to `data` (numpy array or torch tensor) of shape (l, N).
        Uses heuristic jump separation based on extreme returns.
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
        sigma = np.zeros(self.num_channels, dtype=np.float64)
        lamb = np.zeros(self.num_channels, dtype=np.float64)
        p = np.zeros(self.num_channels, dtype=np.float64)
        eta1 = np.zeros(self.num_channels, dtype=np.float64)
        eta2 = np.zeros(self.num_channels, dtype=np.float64)

        for ch in range(self.num_channels):
            series = data[:, ch].astype(np.float64)
            returns = np.diff(np.log(series))

            temp_mu_total = np.mean(returns)
            temp_sigma_total = np.std(returns, ddof=0)

            # Heuristic jump detection
            C = 4.0
            jump_threshold = C * temp_sigma_total
            jumps = np.abs(returns - temp_mu_total) > jump_threshold
            jump_vals = returns[jumps]

            if len(jump_vals) > 1:
                # Separate positive and negative jumps
                pos_jumps = jump_vals[jump_vals > 0]
                neg_jumps = -jump_vals[jump_vals < 0]  # make positive for fitting exponential

                total_time = len(returns)
                lamb[ch] = len(jump_vals) / total_time
                p[ch] = len(pos_jumps) / len(jump_vals) if len(jump_vals) > 0 else 0.5
                eta1[ch] = 1.0 / (np.mean(pos_jumps) + 1e-12) if len(pos_jumps) > 0 else 1.0
                eta2[ch] = 1.0 / (np.mean(neg_jumps) + 1e-12) if len(neg_jumps) > 0 else 1.0
            else:
                # fallback for insufficient jumps
                lamb[ch] = 1e-6
                p[ch] = 0.5
                eta1[ch] = 1.0
                eta2[ch] = 1.0

            # Refine mu and sigma using non-jump returns
            returns_no_jumps = returns[~jumps]

            if len(returns_no_jumps) > 1:
                sigma_diff = np.std(returns_no_jumps, ddof=0)
                mu_diff = np.mean(returns_no_jumps)

                # Expected jump: E[J - 1] = p/eta1 - (1-p)/eta2
                jump_comp_exp = p[ch] / eta1[ch] - (1.0 - p[ch]) / eta2[ch]
                mu[ch] = mu_diff + 0.5 * sigma_diff**2 + lamb[ch] * jump_comp_exp
                sigma[ch] = sigma_diff
            else:
                mu[ch] = temp_mu_total + 0.5 * temp_sigma_total**2
                sigma[ch] = temp_sigma_total

        # Save as torch tensors
        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        self.lamb = torch.tensor(lamb, dtype=torch.float32, device=self.device)
        self.p = torch.tensor(p, dtype=torch.float32, device=self.device)
        self.eta1 = torch.tensor(eta1, dtype=torch.float32, device=self.device)
        self.eta2 = torch.tensor(eta2, dtype=torch.float32, device=self.device)

        return {"mu": self.mu, "sigma": self.sigma, "lamb": self.lamb,
                "p": self.p, "eta1": self.eta1, "eta2": self.eta2}

    def generate(self, num_samples: int, initial_value: Optional[np.ndarray] = None,
        output_length: Optional[int] = None, seed: Optional[int] = None
    ):
        """
        Generate `num_samples` independent sample paths of length `output_length` (or self.length if None)
        for the Kou Double Exponential Jump Diffusion process.

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

        # --- initial values ---
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

        # --- parameters ---
        mu = self.mu.to(device) if self.mu is not None else torch.zeros(N, device=device)
        sigma = self.sigma.to(device) if self.sigma is not None else torch.ones(N, device=device) * 1e-3
        lamb = self.lamb.to(device) if self.lamb is not None else torch.ones(N, device=device) * 1e-6
        p = self.p.to(device) if self.p is not None else torch.full((N,), 0.5, device=device)
        eta1 = self.eta1.to(device) if self.eta1 is not None else torch.ones(N, device=device)
        eta2 = self.eta2.to(device) if self.eta2 is not None else torch.ones(N, device=device)

        # --- expected jump correction ---
        jump_drift_correction = lamb * (p / eta1 - (1 - p) / eta2)

        # --- simulate paths ---
        for k in range(L - 1):
            prev = paths[:, k, :]

            # diffusion increment
            log_drift = (mu - 0.5 * sigma**2 - jump_drift_correction)
            z_diff = torch.randn((num_samples, N), dtype=torch.float32, device=device)
            diffusion_increment = sigma * z_diff

            # Poisson jump arrivals
            poisson_counts = torch.poisson(lamb.unsqueeze(0))  # (num_samples, num_channels)

            # total jumps for each channel
            total_jump = torch.zeros_like(prev)
            for ch in range(N):
                Nj = poisson_counts[:, ch]
                mask = Nj > 0
                if mask.any():
                    for i in mask.nonzero(as_tuple=True)[0]:
                        n_jumps = int(Nj[i].item())
                        if n_jumps > 0:
                            u = torch.rand(n_jumps, device=device)
                            signs = torch.where(u < p[ch], 1.0, -1.0)
                            magnitudes = torch.zeros(n_jumps, device=device)
                            pos_idx = signs > 0
                            neg_idx = signs < 0
                            if pos_idx.any():
                                magnitudes[pos_idx] = torch.distributions.Exponential(eta1[ch]).sample((pos_idx.sum(),))
                            if neg_idx.any():
                                magnitudes[neg_idx] = torch.distributions.Exponential(eta2[ch]).sample((neg_idx.sum(),))
                            total_jump[i, ch] = (signs * magnitudes).sum()

            paths[:, k + 1, :] = prev * torch.exp(log_drift + diffusion_increment + total_jump)

        return paths
