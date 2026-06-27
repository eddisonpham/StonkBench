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


def match_channel_moments(data: torch.Tensor, target: ChannelMomentStats) -> torch.Tensor:
    """Affine per-channel calibration to target mean/std."""
    if data.ndim == 2:
        gen_mean = data.mean(dim=0)
        gen_std = data.std(dim=0, unbiased=True).clamp(min=1e-8)
        return (data - gen_mean) / gen_std * target.std + target.mean
    if data.ndim == 3:
        gen_mean = data.mean(dim=(0, 1))
        gen_std = data.std(dim=(0, 1), unbiased=True).clamp(min=1e-8)
        return (data - gen_mean.view(1, 1, -1)) / gen_std.view(1, 1, -1) * target.std.view(
            1, 1, -1
        ) + target.mean.view(1, 1, -1)
    raise ValueError(f"Expected 2D or 3D tensor, got {tuple(data.shape)}")
