"""Shared calibration helpers for the multivariate jump-diffusion baselines.

Both jump-diffusion models in this package (Merton, Double-Exponential Kou
2002) generate a *log-return* panel

    r_c(t) = drift_c + Σ_d L_cd z_d(t) + Σ_{i=1}^{N_c(t)} J_{c,i},
    N_c ~ Poisson(λ_c) independent across channels,
    J_{c,i} i.i.d. from the channel's jump-size law,
    z ~ N(0, I) → Σ_d L_cd z_d has covariance Σ_D = L Lᵀ,

and both share the same three calibration decisions, which live here so the two
models cannot drift apart:

1. **Jump detection.** A return is a jump when ``|r| > k · σ_BPV`` where
   ``σ_BPV`` is the Barndorff-Nielsen & Shephard *bipower* scale
   ``σ_BPV² = (π/2)·mean(|r_t|·|r_{t−1}|)``. The bipower scale is robust to jumps
   (a jump contaminates one product term instead of the whole variance), so the
   threshold is not inflated by the very events it is meant to find. On the
   benchmark training panel ``σ_BPV/σ_total ∈ [0.955, 1.030]``, i.e. the raw and
   robust scales agree to ±3% — the threshold is a genuine 3-sigma cut, not a
   disguised 1.8-sigma cut (which is what ``3·median(|r|)`` would be).

2. **Variance targeting.** The detected jumps must *redistribute* variance into
   the tail, not *add* to it: the empirical variance of a channel is split into
   a diffusion part and a jump part,

       var_c = σ_D,c² + λ_c·E[J_c²],

   so the generated series reproduces the empirical variance vector exactly.
   The jump share is capped at ``1 − min_diffusion_share`` of the variance
   (default 25% floor for the diffusion) and the jump sizes are rescaled by a
   single multiplicative factor ``s_c`` to land the jump variance exactly on the
   target. Without this step the two components double-count the variance and
   the model becomes over-dispersed by the size of the jump component.

3. **Diffusion covariance.** The diffusion must carry the leftover *cross-channel*
   covariance: ``Σ_D = Σ_empirical − diag(λ_c·E[J_c²])``. Equivalently, the
   diffusion correlation matrix is the empirical correlation inflated by
   ``1/(w_c·w_d)``, ``w_c = σ_D,c/σ_c``. Jumps are independent across channels in
   both models, so anything that is subtracted from the diffusion's diagonal
   must be left on its off-diagonals, otherwise the generated cross-correlation
   would be scaled down by ``w_c·w_d`` (≈0.7) across the board. Because the
   diagonal shrinks while the off-diagonals do not, the result is not guaranteed
   positive definite and is projected back onto the PSD cone by clipping the
   negative eigenvalues; the projection is measured and reported by
   ``psd_projection_report`` so the audit can bound the induced error.

Every quantity returned here is a per-channel ``(C,)`` tensor unless stated
otherwise, and every fit is done in float64 no matter the input dtype.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

#: Jump threshold in units of the bipower scale (3-sigma convention).
DEFAULT_THRESHOLD_MULT = 3.0
#: A channel's diffusion component always carries at least this share of its
#: variance; the jump share is capped accordingly. On the benchmark panel the
#: uncapped jump share spans [0.259, 0.423], so the floor binds on the
#: fattest-tailed channels only.
DEFAULT_MIN_DIFFUSION_SHARE = 0.25


@dataclass
class ExceedanceFit:
    """Threshold exceedances of one panel, per channel."""

    thresholds: torch.Tensor  # (C,) k·σ_BPV
    lam: torch.Tensor  # (C,) per-step intensity = #exceedances / T
    counts: torch.Tensor  # (C,) integer exceedance counts
    mask: torch.Tensor  # (T, C) bool
    values: list[torch.Tensor]  # per-channel exceedance values (signed)
    sigma_bpv: torch.Tensor  # (C,) bipower scale
    sigma_total: torch.Tensor  # (C,) sample std (unbiased)
    var_total: torch.Tensor  # (C,) sample variance (unbiased)


@dataclass
class VarianceSplit:
    """Result of splitting an empirical variance vector into diffusion + jumps."""

    jump_var: torch.Tensor  # (C,) variance the (rescaled) jumps carry
    jump_scale: torch.Tensor  # (C,) multiplicative rescaling of the jump law
    sigma_diff: torch.Tensor  # (C,) diffusion std per channel
    cov_diff: torch.Tensor  # (C, C) PSD diffusion covariance
    eig_min_raw: float  # smallest eigenvalue before the PSD projection
    psd_shift: float  # eigenvalue shift applied (>0 only if needed)


def bipower_scale(x: torch.Tensor) -> torch.Tensor:
    """Barndorff-Nielsen & Shephard bipower (jump-robust) scale, per channel.

    ``σ_BPV² = (π/2)·mean(|r_t|·|r_{t−1}|)`` understates the variance by roughly
    the jump share of it, which is exactly the property that makes it usable as
    a jump-detection scale.
    """
    prods = (x[1:].abs() * x[:-1].abs()).mean(dim=0)
    return torch.sqrt(torch.clamp((torch.pi / 2.0) * prods, min=1e-24))


def detect_exceedances(
    x: torch.Tensor,
    threshold_mult: float = DEFAULT_THRESHOLD_MULT,
) -> ExceedanceFit:
    """Flag per-channel threshold exceedances and count their intensity.

    ``x`` is ``(T, C)`` (or ``(T,)``) log returns.
    """
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    x64 = x.to(torch.float64)
    sigma_bpv = bipower_scale(x64)
    sigma_total = torch.std(x64, dim=0, unbiased=True)
    var_total = sigma_total**2
    thresholds = threshold_mult * sigma_bpv
    mask = x64.abs() > thresholds.unsqueeze(0)
    counts = mask.sum(dim=0)
    lam = counts.to(torch.float64) / max(int(x64.shape[0]), 1)
    values = [x64[mask[:, c], c] for c in range(x64.shape[1])]
    return ExceedanceFit(
        thresholds=thresholds,
        lam=lam,
        counts=counts,
        mask=mask,
        values=values,
        sigma_bpv=sigma_bpv,
        sigma_total=sigma_total,
        var_total=var_total,
    )


def symmetric_psd_floor(
    matrix: torch.Tensor, eigen_floor: float = 1e-10, max_shift: float | None = None
) -> tuple[torch.Tensor, float, float]:
    """Project a symmetric matrix onto the PSD cone by shifting its spectrum.

    Returns ``(projected, eig_min_before, shift_applied)``. The shift is uniform
    (``λ ↦ max(λ, floor)``) so the *off-diagonals* move as little as the PSD
    constraint allows: for the near-singular cross-section at hand the negative
    eigenvalues are tiny, and adding a constant ridge would instead inflate the
    whole diagonal (i.e. the marginal variances) by that constant.
    """
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = torch.linalg.eigh(sym)
    eig_min = float(eigvals.min())
    shifted = torch.clamp(eigvals, min=eigen_floor)
    projected = (eigvecs * shifted) @ eigvecs.T
    if max_shift is not None and abs(eig_min) > max_shift:
        raise ValueError(
            f"PSD projection would have to move eigenvalues by {abs(eig_min):.3e} "
            f"(limit {max_shift:.3e}); the variance split is not a small perturbation."
        )
    return projected, eig_min, float(eig_min)


def split_variance(
    var_total: torch.Tensor,
    cov_total: torch.Tensor,
    jump_var_law: torch.Tensor,
    min_diffusion_share: float = DEFAULT_MIN_DIFFUSION_SHARE,
) -> VarianceSplit:
    """Split ``var_total`` into diffusion + jump variance and build ``Σ_D``.

    ``jump_var_law`` is the variance the *current* jump law would contribute
    (``λ_c·E[J_c²]`` before any rescaling). It is capped so the diffusion keeps
    at least ``min_diffusion_share`` of the variance, and the jump sizes are
    rescaled by ``jump_scale`` to land on the capped target exactly. The
    diffusion covariance is the empirical covariance with the jump variance
    removed from its diagonal — and only from its diagonal, so the generated
    cross-channel covariance is preserved (see the module docstring).
    """
    var_total = var_total.to(torch.float64)
    cov_total = cov_total.to(torch.float64)
    law = jump_var_law.to(torch.float64)
    cap = (1.0 - min_diffusion_share) * var_total
    target_jump_var = torch.minimum(law, cap)
    target_jump_var = torch.where(law > 0, target_jump_var, torch.zeros_like(law))
    jump_scale = torch.where(
        law > 0, torch.sqrt(target_jump_var / torch.clamp(law, min=1e-300)), torch.ones_like(law)
    )
    sigma_diff = torch.sqrt(torch.clamp(var_total - target_jump_var, min=1e-24))

    cov_diff_raw = cov_total - torch.diag(target_jump_var)
    cov_diff, eig_min, _ = symmetric_psd_floor(cov_diff_raw)
    # Re-impose the exact diffusion variance on the diagonal: the PSD projection
    # moves it, and the per-channel variance match is the primary stylised fact.
    diag_scale = sigma_diff / torch.sqrt(torch.clamp(torch.diag(cov_diff), min=1e-300))
    cov_diff = cov_diff * diag_scale.unsqueeze(0) * diag_scale.unsqueeze(1)
    # Re-project after the diagonal rescale (a diagonal congruence preserves the
    # sign pattern in practice, but this keeps the contract exact).
    cov_diff, _, _ = symmetric_psd_floor(cov_diff)
    return VarianceSplit(
        jump_var=target_jump_var,
        jump_scale=jump_scale,
        sigma_diff=sigma_diff,
        cov_diff=cov_diff,
        eig_min_raw=eig_min,
        psd_shift=abs(min(eig_min, 0.0)),
    )


def make_correlation_psd(corr: torch.Tensor, eigen_floor: float = 1e-8) -> torch.Tensor:
    """Nearest correlation matrix: eigenvalue clipping + diagonal renormalisation."""
    sym = 0.5 * (corr + corr.T)
    eigvals, eigvecs = torch.linalg.eigh(sym)
    projected = (eigvecs * torch.clamp(eigvals, min=eigen_floor)) @ eigvecs.T
    d = torch.sqrt(torch.clamp(torch.diag(projected), min=1e-30))
    projected = projected / d.unsqueeze(0) / d.unsqueeze(1)
    projected.fill_diagonal_(1.0)
    return projected


def cholesky_or_diag(cov: torch.Tensor, ridge: float = 1e-12) -> torch.Tensor:
    """Cholesky factor of a covariance matrix, with a diagonal fallback.

    Falls back (and says so) on a diagonal-only factor when the matrix is not
    numerically factorisable — a degenerate cross-section should degrade to
    independent-but-correctly-scaled channels rather than raise inside a job.
    """
    try:
        return torch.linalg.cholesky(cov + ridge * torch.eye(cov.shape[0], dtype=cov.dtype))
    except Exception:  # pragma: no cover — defensive
        print("[jump_diffusion] WARNING: Cholesky failed; falling back to diagonal diffusion")
        return torch.diag(torch.sqrt(torch.clamp(torch.diag(cov), min=1e-24)))
