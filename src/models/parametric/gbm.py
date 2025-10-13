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

        # fitted parameters (per-channel for channels 1..N-1)
        self.mu = None    # torch tensor shape (num_price_channels,)
        self.sigma = None # torch tensor shape (num_price_channels,)

        # store fitted_data and timestamps when fit() is called
        self.fitted_data = None
        self.timestamps = None

    # -------------------
    # --- Estimation ---
    # -------------------
    def fit(self, data):
        """
        Fit GBM parameters to `data` (numpy array or torch tensor) of shape (l, N).

        Estimation is done per-channel (channels 1..N-1) using the MLE for irregular dt:
            Y_i = log(S_{t_{i+1}}) - log(S_{t_i})
            a_hat = sum(Y_i) / sum(dt_i)
            sigma2_hat = (1/N) * sum( (Y_i - a_hat*dt_i)^2 / dt_i )
            mu_hat = a_hat + 0.5 * sigma2_hat

        Saves mu and sigma as torch tensors (on self.device) with length = num_channels-1.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2D array with shape (l, N)")

        l, N = data.shape
        if N != self.num_channels:
            # allow fitting even if the input contains fewer/more channels -- warn or adjust:
            raise ValueError(f"data has {N} channels but model expects {self.num_channels}")

        self.fitted_data = data.copy()
        timestamps = data[:, 0].astype(np.float64)
        if np.any(np.diff(timestamps) < 0):
            raise ValueError("timestamps must be non-decreasing")
        self.timestamps = timestamps

        # compute dt_i for each increment: dt_i = t_{i+1} - t_i, i = 0..(l-2)
        if l > 1:
            dt = np.diff(timestamps)
            # replace any zero dt with a small positive number to avoid division by zero
            dt = np.where(dt <= 0, 1e-8, dt)
        else:
            # not enough points -> default dt = 1.0 (single sample)
            dt = np.array([1.0], dtype=np.float64)

        price_channels = self.num_channels - 1
        mu = np.zeros(price_channels, dtype=np.float64)
        sigma = np.zeros(price_channels, dtype=np.float64)

        for ch in range(price_channels):
            series = data[:, ch + 1].astype(np.float64)

            # Ensure positivity for log; if series contains non-positive values, shift / clip
            safe_series = np.clip(series, 1e-12, None)

            # compute log increments Y_i (length l-1)
            if len(safe_series) < 2:
                # not enough data
                mu[ch] = 0.0
                sigma[ch] = 0.0
                continue

            logp = np.log(safe_series)
            Y = np.diff(logp)  # length l-1

            # check lengths align
            if len(Y) != len(dt):
                # possible if l==1 scenario; ensure shapes
                minlen = min(len(Y), len(dt))
                Y = Y[:minlen]
                dt_local = dt[:minlen]
            else:
                dt_local = dt

            Ninc = len(Y)
            if Ninc == 0:
                mu[ch] = 0.0
                sigma[ch] = 0.0
                continue

            # MLE: a_hat = sum(Y) / sum(dt)
            sum_dt = float(np.sum(dt_local))
            a_hat = float(np.sum(Y)) / sum_dt if sum_dt > 0 else 0.0

            # sigma^2_hat = (1/N) * sum( (Y - a_hat*dt)^2 / dt )
            # avoid divide-by-zero (we clipped dt_small earlier)
            resid = Y - a_hat * dt_local
            # compute weighted residual sum
            weighted_squared = np.sum((resid ** 2) / dt_local)
            sigma2_hat = (weighted_squared / Ninc) if Ninc > 0 else 0.0
            # ensure non-negative and numerical stability
            sigma2_hat = max(sigma2_hat, 0.0)
            sigma_hat = float(np.sqrt(sigma2_hat))

            mu_hat = a_hat + 0.5 * sigma2_hat

            mu[ch] = mu_hat
            sigma[ch] = sigma_hat

        # Save as torch tensors on device
        self.mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)

        # keep fitted timestamps/dt for generate
        self._fitted_dt = dt  # numpy array of length l-1 (if l>1)
        return {"mu": self.mu, "sigma": self.sigma}

    # -----------------------
    # --- Sample / Generate ---
    # -----------------------
    def _get_dt_sequence(self, l_out: int):
        """
        Return a dt sequence of length l_out (dt for steps between samples).

        If we have fitted timestamps of length Lfitted, we build a dt_seq of length l_out-1
        by (a) using the fitted per-step dt where possible, and (b) using the mean dt
        or repeating the last dt to fit the requested length.

        Returns: numpy array length l_out-1 (if l_out > 1)
        """
        if l_out <= 1:
            return np.array([1.0], dtype=np.float32)

        if hasattr(self, "timestamps") and self.timestamps is not None and len(self.timestamps) > 1:
            fitted_dt = np.diff(self.timestamps).astype(np.float64)
            # if the requested length matches the fitted length, use fitted dt
            if len(fitted_dt) == l_out - 1:
                return fitted_dt.astype(np.float32)
            # else tile/interpolate: simplest is to repeat or use mean
            mean_dt = float(np.mean(fitted_dt)) if len(fitted_dt) > 0 else 1.0
            # build dt_seq by taking fitted_dt repeated to reach l_out-1
            # if fitted shorter, repeat last; if longer, take prefix
            if len(fitted_dt) >= l_out - 1:
                return fitted_dt[: l_out - 1].astype(np.float32)
            else:
                # repeat last dt to fill
                pad = np.full((l_out - 1 - len(fitted_dt),), fitted_dt[-1] if len(fitted_dt)>0 else mean_dt, dtype=np.float32)
                return np.concatenate([fitted_dt.astype(np.float32), pad]).astype(np.float32)
        else:
            # no fitted timestamps -> uniform dt of 1.0
            return np.ones(l_out - 1, dtype=np.float32)

    def generate(self, num_samples: int, initial_price: Optional[np.ndarray] = None, output_length: Optional[int] = None, seed: Optional[int] = None):
        """
        Generate `num_samples` independent sample paths of length `output_length` (or self.length if None).

        Returns a torch tensor of shape (num_samples, output_length, num_channels).
        Channel 0: timestamps (constructed from fitted timestamps if available, else 0..L-1)
        Channels 1..N-1: simulated channels.

        OHLC handling assumptions:
          - If the input channels are interpreted as [Open, Close, High, Low, Volume] (or subset),
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

        # dt sequence for the L-1 steps
        dt_seq = self._get_dt_sequence(L)  # numpy length L-1
        dt_torch = torch.tensor(dt_seq, dtype=torch.float32, device=device)  # length L-1

        # build timestamps (channel 0)
        if hasattr(self, "timestamps") and self.timestamps is not None and len(self.timestamps) == L:
            timestamps = torch.tensor(self.timestamps, dtype=torch.float32, device=device)
        else:
            # if we have fitted timestamps but different length, build cumulative from dt_seq
            # start at t0 = 0
            t0 = 0.0
            ts = [t0]
            for d in dt_seq:
                ts.append(ts[-1] + float(d))
            timestamps = torch.tensor(np.array(ts, dtype=np.float32)[:L], dtype=torch.float32, device=device)

        # allocate output
        paths = torch.zeros((num_samples, L, N), dtype=torch.float32, device=device)
        # fill timestamps
        paths[:, :, 0] = timestamps.unsqueeze(0).expand(num_samples, -1)

        # initial prices
        if initial_price is None:
            # if we have fitted_data, take first observed values
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

        # set t=0 values
        paths[:, 0, 1:] = torch.tensor(init_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(num_samples, -1)

        # If parameters not fitted, fallback to zeros/small sigma
        if self.mu is None or self.sigma is None:
            mu = torch.zeros((price_channels,), dtype=torch.float32, device=device)
            sigma = torch.zeros((price_channels,), dtype=torch.float32, device=device)
        else:
            mu = self.mu.to(device)
            sigma = self.sigma.to(device)

        # Simulate per-step (vectorized)
        # We interpret dt_torch[k] as the time increment from step k to k+1 (for k in 0..L-2)
        for k in range(L - 1):
            dtk = float(dt_seq[k])
            # sample independent normals: shape (num_samples, price_channels)
            z = torch.randn((num_samples, price_channels), dtype=torch.float32, device=device)

            # drift and diffusion scaled for dt
            # S_{t+dt} = S_t * exp( (mu - 0.5*sigma^2) * dt + sigma * sqrt(dt) * z )
            drift = (mu - 0.5 * sigma * sigma) * dtk  # shape (price_channels,)
            diffusion = sigma * (np.sqrt(dtk)) * z    # broadcast: (num_samples, price_channels)

            prev = paths[:, k, 1:]  # shape (num_samples, price_channels)
            # compute multiplicative factor and update
            factor = torch.exp(drift.unsqueeze(0) + diffusion)  # (num_samples, price_channels)
            new_price = prev * factor
            paths[:, k + 1, 1:] = new_price

            # If we want OHLC semantics (assuming channels are [Open, Close, High, Low, Volume] or similar),
            # we can post-process after simulating entire Close series. Simpler approach: set:
            #   Open_{t+1} = Close_t
            #   Close_{t+1} = new_price
            #   High/Low = max/min(Open,Close)
            # Here we check if channels seem like OHLCV by count or user convention; for generality we leave channels as-is.
            # If user expects OHLC: they should map channels accordingly after generation.

        # Optional: interpret common OHLC layout (if exactly 5 channels beyond timestamp)
        # Layout example: channels 1..5 = [Open, Close, High, Low, Volume]
        if price_channels >= 4:
            # transform simulated series into consistent OHLC:
            # We'll take the simulated 'Close' as channel index 2 (i.e., channel number 2 overall),
            # but since channel ordering is user-defined, leave it flexible: we'll assume channel indices:
            # 1: Open, 2: Close, 3: High, 4: Low, 5: Volume (if present)
            # Implementation below assumes that the user provided channels in that order.
            # We'll set:
            #   Open_{t} = Close_{t-1}
            #   High_{t} = max(Open_t, Close_t)
            #   Low_{t} = min(Open_t, Close_t)
            # volumes (if present) remain as simulated channel (or could be replaced)
            # Note: This is a simple consistent construction, not adding intraday randomness.
            try:
                # indices relative to channel axis (0-based): open=1, close=2, high=3, low=4, volume=5
                # Check we have these indices
                if price_channels >= 2:
                    # set Open for t>0 as previous Close
                    open_idx = 1
                    close_idx = 2
                    high_idx = 3 if price_channels >= 3 else None
                    low_idx = 4 if price_channels >= 4 else None

                    # ensure we have a close channel to base opens off
                    if close_idx < price_channels + 1:
                        # first Open stays as initial (already set), subsequent opens are previous closes
                        # note: channels in paths are 1..N-1; we offset by 1 when indexing
                        for t in range(1, L):
                            paths[:, t, open_idx] = paths[:, t - 1, close_idx]
                            # set high/low if available
                            if high_idx is not None and low_idx is not None:
                                c = paths[:, t, close_idx]
                                o = paths[:, t, open_idx]
                                paths[:, t, high_idx] = torch.maximum(c, o)
                                paths[:, t, low_idx] = torch.minimum(c, o)
            except Exception:
                # if anything goes wrong, skip OHLC formatting and keep raw simulated channels
                pass

        return paths

