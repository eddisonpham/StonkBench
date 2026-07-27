"""Train-set moment matching for adapter outputs (integration layer only)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ChannelMomentStats:
    mean: torch.Tensor  # (C,)
    std: torch.Tensor  # (C,)

    @classmethod
    def from_windows(cls, windows: torch.Tensor) -> "ChannelMomentStats":
        if windows.ndim != 3:
            raise ValueError("Expected train windows shaped (N, L, C)")
        mean = windows.mean(dim=(0, 1))
        std = windows.std(dim=(0, 1), unbiased=True).clamp(min=1e-8)
        return cls(mean=mean.float(), std=std.float())

    @classmethod
    def from_univariate(cls, series: torch.Tensor) -> "ChannelMomentStats":
        if series.ndim != 2:
            raise ValueError("Expected series shaped (N, L)")
        mean = series.mean().reshape(1)
        std = series.std(unbiased=True).clamp(min=1e-8).reshape(1)
        return cls(mean=mean.float(), std=std.float())


def shuffle_time_within_windows(data: torch.Tensor, seed: int) -> torch.Tensor:
    """Remove spurious within-window autocorrelation (preserves per-lag marginals)."""
    if data.ndim != 3:
        raise ValueError(f"Expected 3D tensor (N, L, C), got {tuple(data.shape)}")
    out = data.clone()
    generator = torch.Generator(device=data.device)
    generator.manual_seed(seed)
    for i in range(out.shape[0]):
        perm = torch.randperm(out.shape[1], generator=generator, device=data.device)
        out[i] = out[i, perm, :]
    return out


def match_channel_moments(
    data: torch.Tensor,
    target: ChannelMomentStats,
    min_std_ratio: float = 0.25,
) -> torch.Tensor:
    """Affine per-channel calibration to target mean/std.

    Parameters
    ----------
    data
        (N, L) or (N, L, C) generated tensor to be calibrated.
    target
        ``ChannelMomentStats`` carrying the per-channel target mean / std
        (typically computed from training windows).
    min_std_ratio
        Floor for the effective generated std as a **fraction of
        ``target.std``**.  When the generator's per-channel std collapses
        below this ratio, the divisor is held at ``min_std_ratio * target.std``
        instead of the raw tiny ``gen_std`` (which would otherwise be clamped
        at ``1e-8`` and act as an arbitrary amplification factor of
        ``target.std / 1e-8 ≈ 1000×``).  Default ``0.25`` keeps the effective
        scale at most ``4×`` the target std, which is enough to recover a
        realistic dispersion on collapsed channels without the sign-bias
        amplification pathology observed when ``min_std_ratio == 0``.

        Set to ``0.0`` to fall back to the legacy behaviour.  Set to higher
        (e.g. ``0.5``) for more conservative calibration on badly-collapsed
        generators (PCF-GAN's signature-Wasserstein critic is prone to
        variance collapse, which is what this knob was added to address).
    """
    if min_std_ratio < 0:
        raise ValueError(f"min_std_ratio must be >= 0, got {min_std_ratio}")
    if data.ndim == 2:
        gen_mean = data.mean(dim=0)
        gen_std = data.std(dim=0, unbiased=True)
        # Two-tier floor: (1) ratio of target.std guards against the
        # amplification pathology described in the docstring; (2) absolute
        # 1e-4 guards against catastrophic `target.std == 0` edge case.
        effective_std = torch.clamp(
            gen_std, min=max(min_std_ratio * target.std, torch.full_like(target.std, 1e-4))
        )
        return (data - gen_mean) / effective_std * target.std + target.mean
    if data.ndim == 3:
        gen_mean = data.mean(dim=(0, 1))
        gen_std = data.std(dim=(0, 1), unbiased=True)
        effective_std = torch.clamp(
            gen_std, min=torch.maximum(min_std_ratio * target.std, torch.full_like(target.std, 1e-4))
        )
        return (
            (data - gen_mean.view(1, 1, -1)) / effective_std.view(1, 1, -1)
            * target.std.view(1, 1, -1) + target.mean.view(1, 1, -1)
        )
    raise ValueError(f"Expected 2D or 3D tensor, got {tuple(data.shape)}")
