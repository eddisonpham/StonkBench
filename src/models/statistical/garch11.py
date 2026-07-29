"""Multivariate GARCH(1,1) — frozen-correlation (Cholesky-of-residual-correlation).

Per-channel: independent estimate of (μ, ω, α, β) via the `arch` library GARCH(1,1)
specification (identical to the original univariate GARCH11 model). Each channel has
its own conditional variance σ²_t persistence (volatility clustering) — preserves
stylized fact of per-channel ARCH effects.

Cross-channel: at generate-time, after drawing independent standard normals, we
correlate them through the Cholesky factor of the empirical correlation matrix of the
standardized residuals z_t = (x_t − μ_c) / σ_t. This yields a *frozen* multivariate
GARCH(1,1) — per-channel vol persistence + static cross-channel correlation.

The proper multivariate GARCH with time-varying correlation is Dynamic Conditional
Correlation (DCC, Engle 2002); that requires a 2D log-likelihood optimization over
(α_dcc, β_dcc) per the DCC recursion Q_t = (1-α-β) Q̄ + α z_{t−1} z_{t−1}' + β Q_{t−1}.
We use the simpler "static-correlation" variant here for clarity and fold-stable
unit testing. Flagged as future-work for DCC upgrade.
"""
from __future__ import annotations

import numpy as np
import torch
from arch import arch_model

from src.models.base.base_model import StatisticalModel


class GARCH11(StatisticalModel):
    def __init__(self):
        super().__init__()
        self.mu = None
        self.omega = None
        self.alpha = None
        self.beta = None
        self.chol_factor = None  # (C, C) Cholesky of standardized residuals' cross-channel correlation
        self.num_channels = 0

    def fit(self, log_returns: torch.Tensor) -> None:
        data = log_returns
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        if data.ndim != 2:
            raise ValueError(
                f"GARCH11 expects input shaped (L,) or (L, C), got {tuple(log_returns.shape)}"
            )

        self.num_channels = data.shape[1]
        mu_vals: list[float] = []
        omega_vals: list[float] = []
        alpha_vals: list[float] = []
        beta_vals: list[float] = []
        standardized_residuals: list[np.ndarray] = []

        for c in range(self.num_channels):
            channel_np = data[:, c].detach().cpu().numpy()
            am = arch_model(
                channel_np,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
                rescale=False,
            )
            model_fit = am.fit(disp="off")
            mu_vals.append(float(model_fit.params["mu"]))
            omega_vals.append(float(model_fit.params["omega"]))
            alpha_vals.append(float(model_fit.params["alpha[1]"]))
            beta_vals.append(float(model_fit.params["beta[1]"]))
            # arch exposes standardized residuals via std_resid (resid / conditional_volatility).
            if hasattr(model_fit, "std_resid") and model_fit.std_resid is not None:
                z = np.asarray(model_fit.std_resid, dtype=np.float32)
            else:
                z = np.asarray(
                    model_fit.resid / model_fit.conditional_volatility, dtype=np.float32
                )
            standardized_residuals.append(z)

        self.mu = torch.tensor(mu_vals, dtype=torch.float32)
        self.omega = torch.tensor(omega_vals, dtype=torch.float32)
        self.alpha = torch.tensor(alpha_vals, dtype=torch.float32)
        self.beta = torch.tensor(beta_vals, dtype=torch.float32)

        # Multivariate correlation: from standardized residuals z_t (T, C).
        if self.num_channels > 1:
            z_mat = torch.tensor(np.stack(standardized_residuals, axis=1), dtype=torch.float32)
            corr = torch.corrcoef(z_mat.T)
        else:
            corr = torch.tensor([[1.0]], dtype=torch.float32)
        # Tiny ridge → ensure PSD on near-degenerate residual cross-sections.
        corr = corr + 1e-6 * torch.eye(self.num_channels, dtype=corr.dtype)
        try:
            self.chol_factor = torch.linalg.cholesky(corr.to(torch.float64)).to(torch.float32)
        except Exception:
            # Fallback: identity correlation (independent channels — at least per-channel vol persists).
            self.chol_factor = torch.eye(self.num_channels, dtype=torch.float32)
        print(
            f"GARCH11 (multivariate frozen-corr) fitted with {self.num_channels} channel(s); "
            f"Cholesky derived from standardized-residual correlation matrix"
        )

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.chol_factor is None or any(
            v is None for v in [self.mu, self.omega, self.alpha, self.beta]
        ):
            raise RuntimeError("Call fit() before generate().")

        log_returns = torch.zeros(
            (num_samples, generation_length, self.num_channels), dtype=self.mu.dtype
        )
        sigma2 = torch.zeros(
            (num_samples, generation_length, self.num_channels), dtype=self.mu.dtype
        )
        epsilon = torch.zeros(
            (num_samples, generation_length, self.num_channels), dtype=self.mu.dtype
        )

        # Stationary long-run variance per channel: ω / (1 - α - β).
        denom = torch.clamp(1 - self.alpha - self.beta, min=1e-8)
        sigma2[:, 0, :] = self.omega.unsqueeze(0) / denom.unsqueeze(0)

        # First-step innovation: independent draws correlated via Cholesky.
        z0_indep = torch.randn(num_samples, self.num_channels, dtype=self.mu.dtype)
        z0_corr = torch.einsum("rk,kc->rc", z0_indep, self.chol_factor)
        sigma_t0 = torch.sqrt(torch.clamp(sigma2[:, 0, :], min=1e-12))
        epsilon[:, 0, :] = sigma_t0 * z0_corr
        log_returns[:, 0, :] = self.mu.unsqueeze(0) + epsilon[:, 0, :]

        for t in range(1, generation_length):
            sigma2[:, t, :] = (
                self.omega.unsqueeze(0)
                + self.alpha.unsqueeze(0) * epsilon[:, t - 1, :] ** 2
                + self.beta.unsqueeze(0) * sigma2[:, t - 1, :]
            )
            zt_indep = torch.randn(num_samples, self.num_channels, dtype=self.mu.dtype)
            zt_corr = torch.einsum("rk,kc->rc", zt_indep, self.chol_factor)
            sigma_t = torch.sqrt(torch.clamp(sigma2[:, t, :], min=1e-12))
            epsilon[:, t, :] = sigma_t * zt_corr
            log_returns[:, t, :] = self.mu.unsqueeze(0) + epsilon[:, t, :]

        if self.num_channels == 1:
            return log_returns.squeeze(-1)
        return log_returns
