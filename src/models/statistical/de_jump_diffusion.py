"""Multivariate Double-Exponential Jump-Diffusion (Kou 2002).

Per-channel: (λ, p, η₁, η₂) estimated by threshold jump detection on the panel's
log returns, with the diffusion σ backed out by **variance targeting** — the
same shared calibration as the Merton baseline (``jump_diffusion_utils``).

Jump law (the point of this model relative to Merton's Gaussian jumps): a jump is
positive with probability ``p`` and then has size ``u + Exp(η₁)``, negative with
probability ``1 − p`` and size ``−(u + Exp(η₂))``, i.e. the *peaks-over-threshold*
reading of "a jump is an exceedance of ``u = k·σ_BPV``", fitted on the excesses
``|r| − u`` rather than on ``|r|`` itself. Exponential jump sizes give the
asymmetric, heavier-than-Gaussian tails that daily equity returns exhibit, which
a Gaussian jump law cannot reach at the calibrated intensity.

As in the classical multivariate jump-diffusion, jumps are independent across
channels and the *diffusion* carries the whole cross-channel covariance
(``Σ_D = Σ_empirical − diag(λ_c·E[J_c²])``), so the generated panel reproduces
the empirical mean vector, variance vector and covariance matrix.

References
----------
Kou, S. G. (2002), "A jump-diffusion model for option pricing", Management
Science 48(8), 1086-1101 (double-exponential jump sizes).
Kou & Wang (2004), "Option pricing under a double exponential jump diffusion
model", Management Science 50(9), 1178-1192.
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

#: Fallback exponential rate when a side of the exceedance sample is empty
#: (≈ an 8-sigma mean jump size); such a channel has no jumps on that side.
_NULL_ETA = 8.0


class DoubleExponentialJumpDiffusion(StatisticalModel):
    def __init__(self, threshold_mult: float = DEFAULT_THRESHOLD_MULT):
        super().__init__()
        self.threshold_mult = float(threshold_mult)
        self.mean = None  # target per-step mean of the generated log returns
        self.drift = None  # mean − λ·E[J]
        self.sigma = None  # diffusion std σ_D per channel
        self.lam = None
        self.p = None  # P(jump > 0)
        self.eta1 = None  # positive jump rate (post-rescale)
        self.eta2 = None  # negative jump rate (post-rescale)
        self.shift = None  # rescaled threshold s·u, the minimum |jump|
        self.kappa = None  # E[e^J] − 1 under the *scaled* law (reference only)
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
            raise ValueError(f"DEJD expects input shaped (L,) or (C,), got {tuple(log_returns.shape)}")

        x = data.to(torch.float64)
        self.num_channels = x.shape[1]
        C = self.num_channels

        exc = detect_exceedances(x, self.threshold_mult)

        # ---- double-exponential law fitted on the *excesses* ---------------
        p = torch.zeros(C, dtype=torch.float64)
        eta1 = torch.full((C,), _NULL_ETA, dtype=torch.float64)
        eta2 = torch.full((C,), _NULL_ETA, dtype=torch.float64)
        for c in range(C):
            j = exc.values[c]
            if j.numel() == 0:
                p[c] = 0.5
                continue
            pos = j[j > 0] - exc.thresholds[c]
            neg = -j[j < 0] - exc.thresholds[c]
            p[c] = pos.numel() / j.numel()
            if pos.numel() > 0 and pos.mean() > 0:
                eta1[c] = 1.0 / pos.mean()  # MLE for an exponential on the excesses
            if neg.numel() > 0 and neg.mean() > 0:
                eta2[c] = 1.0 / neg.mean()

        # Moments of the *unscaled* sampling law
        #     J = +(u + Exp(η₁)) w.p. p,   J = −(u + Exp(η₂)) w.p. 1−p
        # E[(u+Y)²] = (u + 1/η)² + 1/η²  for Y ~ Exp(η).
        up = exc.thresholds
        m_pos = up + 1.0 / eta1
        m_neg = up + 1.0 / eta2
        m2_pos = m_pos**2 + 1.0 / eta1**2
        m2_neg = m_neg**2 + 1.0 / eta2**2
        e_jump = p * m_pos - (1.0 - p) * m_neg
        e_jump_sq = p * m2_pos + (1.0 - p) * m2_neg
        jump_var_law = exc.lam * e_jump_sq

        # ---- variance targeting -------------------------------------------
        cov_total = torch.cov(x.T) if C > 1 else torch.var(x[:, 0], unbiased=True).reshape(1, 1)
        split = split_variance(
            var_total=exc.var_total,
            cov_total=cov_total,
            jump_var_law=jump_var_law,
            min_diffusion_share=DEFAULT_MIN_DIFFUSION_SHARE,
        )

        # Rescaling the jump size by s scales both moments (E[J] by s, E[J²] by
        # s²); on the law this is η ↦ η/s and u ↦ s·u, used at generate time.
        s = split.jump_scale
        self.lam = exc.lam.to(torch.float32)
        self.p = p.to(torch.float32)
        self.eta1 = (eta1 / s).to(torch.float32)
        self.eta2 = (eta2 / s).to(torch.float32)
        self.shift = (s * up).to(torch.float32)
        self.jump_var = split.jump_var.to(torch.float32)
        self.sigma = split.sigma_diff.to(torch.float32)
        self.threshold = exc.thresholds.to(torch.float32)
        self.mean = x.mean(dim=0).to(torch.float32)
        self.drift = (x.mean(dim=0) - exc.lam * s * e_jump).to(torch.float32)
        # Reference-only: the SDE-level compensation E[e^J] − 1 for the scaled law.
        e_exp_pos = torch.exp(self.shift.to(torch.float64)) * self.eta1.to(torch.float64) / torch.clamp(
            self.eta1.to(torch.float64) - 1.0, min=1e-6
        )
        e_exp_neg = torch.exp(-self.shift.to(torch.float64)) * self.eta2.to(torch.float64) / (
            self.eta2.to(torch.float64) + 1.0
        )
        self.kappa = (self.p.to(torch.float64) * e_exp_pos + (1.0 - self.p.to(torch.float64)) * e_exp_neg - 1.0).to(
            torch.float32
        )

        self.cov_diff = split.cov_diff.to(torch.float32)
        self.chol_factor = cholesky_or_diag(split.cov_diff).to(torch.float32)

        jump_share = (split.jump_var / torch.clamp(exc.var_total, min=1e-300)).mean()
        print(
            f"DEJD (multivariate) fitted with {C} channel(s); "
            f"threshold={self.threshold_mult:.1f}·σ_BPV, "
            f"λ={float(self.lam.mean()):.5f}/step ({float(self.lam.mean()) * 252:.1f}/yr), "
            f"p={float(self.p.mean()):.3f}, "
            f"η₁={float(self.eta1.mean()):.1f}, η₂={float(self.eta2.mean()):.1f}, "
            f"jump share of variance={float(jump_share):.3f}, "
            f"Σ_D PSD shift={split.psd_shift:.2e}"
        )

    def generate(self, num_samples: int, generation_length: int, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.chol_factor is None or any(
            v is None
            for v in [self.mean, self.drift, self.sigma, self.lam, self.p, self.eta1, self.eta2, self.shift]
        ):
            raise RuntimeError("Call fit() before generate().")

        # Correlated diffusion: chol_factor is LOWER-triangular with Σ_D = L Lᵀ,
        # so the draw is z @ Lᵀ. The covariance (hence the per-channel variance)
        # is already carried by L; an extra per-channel σ factor would scale it
        # down by σ² and leave the diffusion with almost no variance.
        z_indep = torch.randn(num_samples, generation_length, self.num_channels, dtype=self.drift.dtype)
        diffusion = z_indep @ self.chol_factor.T

        # Independent double-exponential jumps, drawn in closed form:
        # J = ±(s·u + Exp(η/s)) with sign probability p.
        jumps = torch.empty_like(diffusion)
        for c in range(self.num_channels):
            lam_c = float(self.lam[c])
            counts = torch.poisson(
                torch.full((num_samples, generation_length), lam_c, dtype=self.drift.dtype)
            )
            u = torch.rand(num_samples, generation_length, dtype=self.drift.dtype)
            exp_u = torch.clamp(-torch.log(torch.clamp(1.0 - u, min=1e-12)), min=0.0)
            pos = self.shift[c] + exp_u / self.eta1[c]
            neg = self.shift[c] + exp_u / self.eta2[c]
            sign = torch.rand(num_samples, generation_length, dtype=self.drift.dtype) < self.p[c]
            size = torch.where(sign, pos, -neg)
            jumps[:, :, c] = counts * size

        log_returns = self.drift.view(1, 1, -1) + diffusion + jumps

        if self.num_channels == 1:
            return log_returns.squeeze(-1)
        return log_returns
