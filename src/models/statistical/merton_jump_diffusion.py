"""Multivariate Merton Jump-Diffusion.

Per-channel: (λ, μ_J, σ_J) estimated by threshold jump detection on the panel's
log returns, with the diffusion σ backed out by **variance targeting** — see
``jump_diffusion_utils`` for the detection scale (bipower), the variance split
and the reason each step exists. In short, per channel

    var_c = σ_D,c² + λ_c·E[J_c²],   E[r_c] = mean_c,

so the generated panel reproduces the empirical mean *and* variance vectors
exactly, and the jumps redistribute tail mass instead of adding variance.

Cross-channel: the ``(R, L, C)`` standard-normal tensor is multiplied by the
Cholesky factor of the *diffusion* covariance ``Σ_D = Σ_empirical −
diag(λ_c E[J_c²])``, so the diffusion carries the whole cross-channel covariance
while the Poisson jumps stay independent per channel (independent-increment
assumption, as in the classical multivariate Merton model). Generated total
covariance = ``Σ_D + diag(jump variance) = Σ_empirical``.

Output: (R, L, C) for multivariate, (R, L) for univariate.

References
----------
Merton, R. C. (1976), "Option pricing when underlying stock returns are
discontinuous", JFE 3(1-2), 125-144.
Barndorff-Nielsen & Shephard (2004), "Power and bipower variation with
stochastic volatility and jumps", J. Financial Econometrics 2(1), 1-37
(threshold-detection scale).
"""
from __future__ import annotations

import torch
import numpy as np

from src.models.base.base_model import StatisticalModel
from src.models.statistical.jump_diffusion_utils import (
    DEFAULT_MIN_DIFFUSION_SHARE,
    DEFAULT_THRESHOLD_MULT,
    cholesky_or_diag,
    detect_exceedances,
    split_variance,
)


class MertonJumpDiffusion(StatisticalModel):
    def __init__(self, threshold_mult: float = DEFAULT_THRESHOLD_MULT):
        super().__init__()
        self.threshold_mult = float(threshold_mult)
        self.mean = None  # target per-step mean of the generated log returns
        self.drift = None  # mean − λ·E[J]
        self.sigma = None  # diffusion std σ_D per channel
        self.lam = None
        self.mu_j = None
        self.sigma_j = None
        self.kappa = None  # exp(μ_J + σ_J²/2) − 1 (SDE level compensation, reference only)
        self.chol_factor = None  # (C, C), Σ_D = L Lᵀ
        self.cov_diff = None
        self.jump_var = None
        self.threshold = None
        self.num_channels = 0

    def fit(self, log_returns: torch.Tensor) -> None:
        data = log_returns
        if data.ndim == 1:
            data = data.unsqueeze(-1)
        if data.ndim != 2:
            raise ValueError(f"Merton expects input shaped (L,) or (C,), got {tuple(log_returns.shape)}")

        x = data.to(torch.float64)
        self.num_channels = x.shape[1]
        C = self.num_channels

        # ---- threshold jump detection + intensity --------------------------
        exc = detect_exceedances(x, self.threshold_mult)

        # ---- Gaussian jump-size law, fitted on the exceedances ------------
        # Method of moments on the exceedance sample. Because the jump sizes are
        # generated from exactly this fitted law, the first two moments used for
        # variance targeting below are the *observed* exceedance moments
        # (μ_J² + σ_J² == mean(exceedance²)), so the marginal variance match is
        # exact rather than approximately restored. What the Gaussian assumption
        # buys is the shape of the jump distribution only: the real exceedance
        # sample is not Gaussian, so the generated tails above the threshold are
        # lighter than the empirical ones — the same limitation the DEJD
        # baseline exists to fix (its double-exponential jump sizes are fitted
        # for exactly this reason).
        mu_j = torch.zeros(C, dtype=torch.float64)
        sigma_j = torch.zeros(C, dtype=torch.float64)
        for c in range(C):
            j = exc.values[c]
            if j.numel() > 0:
                mu_j[c] = j.mean()
                sigma_j[c] = torch.clamp(j.std(unbiased=True), min=1e-12)

        # E[J²] for a Gaussian jump: μ_J² + σ_J².
        jump_var_law = exc.lam * (mu_j**2 + sigma_j**2)

        # ---- variance targeting -------------------------------------------
        cov_total = torch.cov(x.T) if C > 1 else torch.var(x[:, 0], unbiased=True).reshape(1, 1)
        split = split_variance(
            var_total=exc.var_total,
            cov_total=cov_total,
            jump_var_law=jump_var_law,
            min_diffusion_share=DEFAULT_MIN_DIFFUSION_SHARE,
        )

        # Rescale the jump law by s (μ_J and σ_J both scale with s, so E[J²]
        # scales by s² and lands exactly on the targeted jump variance).
        s = split.jump_scale
        self.mu_j = (mu_j * s).to(torch.float32)
        self.sigma_j = (sigma_j * s).to(torch.float32)
        self.lam = exc.lam.to(torch.float32)
        self.jump_var = split.jump_var.to(torch.float32)
        self.sigma = split.sigma_diff.to(torch.float32)
        self.threshold = exc.thresholds.to(torch.float32)
        # Drift absorbs the expected jump so E[r] = mean(x) exactly.
        self.mean = x.mean(dim=0).to(torch.float32)
        self.drift = (x.mean(dim=0) - self.lam.to(torch.float64) * self.mu_j.to(torch.float64)).to(
            torch.float32
        )
        self.kappa = (torch.exp(self.mu_j.to(torch.float64) + 0.5 * self.sigma_j.to(torch.float64) ** 2) - 1.0).to(
            torch.float32
        )

        self.cov_diff = split.cov_diff.to(torch.float32)
        self.chol_factor = cholesky_or_diag(split.cov_diff).to(torch.float32)

        jump_share = (split.jump_var / torch.clamp(exc.var_total, min=1e-300)).mean()
        print(
            f"Merton (multivariate) fitted with {C} channel(s); "
            f"threshold={self.threshold_mult:.1f}·σ_BPV, "
            f"λ={float(self.lam.mean()):.5f}/step ({float(self.lam.mean()) * 252:.1f}/yr), "
            f"jump share of variance={float(jump_share):.3f}, "
            f"Σ_D PSD shift={split.psd_shift:.2e}"
        )

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.chol_factor is None or any(
            v is None
            for v in [self.mean, self.drift, self.sigma, self.lam, self.mu_j, self.sigma_j]
        ):
            raise RuntimeError("Call fit() before generate().")

        # Correlated diffusion. chol_factor is LOWER-triangular with Σ_D = L Lᵀ,
        # so the draw is z @ Lᵀ (i.e. L z per row). The covariance is already
        # carried by L — multiplying by a further per-channel σ here would
        # *double scale* the shock and shrink the diffusion variance by σ².
        z_indep = torch.randn(num_samples, generation_length, self.num_channels, dtype=self.drift.dtype)
        diffusion = z_indep @ self.chol_factor.T

        # Per-channel independent compound-Poisson jumps, drawn in closed form:
        # Σ_{i=1}^{N} J_i = N·μ_J + sqrt(N)·σ_J·z for Gaussian J.
        jumps = torch.empty_like(diffusion)
        for c in range(self.num_channels):
            counts = torch.poisson(
                torch.full(
                    (num_samples, generation_length), float(self.lam[c]), dtype=self.drift.dtype
                )
            )
            jumps[:, :, c] = counts * self.mu_j[c] + self.sigma_j[c] * torch.sqrt(counts) * torch.randn(
                num_samples, generation_length, dtype=self.drift.dtype
            )

        log_returns = self.drift.view(1, 1, -1) + diffusion + jumps

        if self.num_channels == 1:
            return log_returns.squeeze(-1)
        return log_returns
