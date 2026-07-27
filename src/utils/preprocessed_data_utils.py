"""Helpers for loading model-ready preprocessed tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.experiments.core.contracts import StandardBatch
from src.utils.env import get_dl_set_path, get_stats_set_path


def resolve_dl_set_path() -> str:
    """Return the canonical dl_set.pt path."""
    return str(get_dl_set_path())


def resolve_stats_set_path() -> str:
    """Return the canonical statsmodel_set.pt path."""
    return str(get_stats_set_path())


def channel_norm_stats(dl_set: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (mean, std) if the DL set was z-scored during preprocessing."""
    mean = dl_set.get("channel_mean")
    std = dl_set.get("channel_std")
    if mean is None or std is None:
        return None
    return mean.float(), std.float()


def denormalize_channels(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Map z-scored windows back to raw feature space."""
    mean = mean.to(device=data.device, dtype=data.dtype)
    std = std.to(device=data.device, dtype=data.dtype)
    if data.ndim == 2:
        return data * std + mean
    if data.ndim == 3:
        return data * std.view(1, 1, -1) + mean.view(1, 1, -1)
    raise ValueError(f"Expected 2D or 3D tensor, got {tuple(data.shape)}")


def normalize_channels(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if data.ndim == 2:
        return (data - mean) / std
    if data.ndim == 3:
        return (data - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    raise ValueError(f"Expected 2D or 3D tensor, got {tuple(data.shape)}")


def sliding_window_2d(series: torch.Tensor, window_size: int, stride: int = 1) -> torch.Tensor:
    if series.ndim != 2:
        raise ValueError(f"Expected 2D tensor (T, C), got {tuple(series.shape)}")
    if series.shape[0] < window_size:
        return torch.empty((0, window_size, series.shape[1]), dtype=series.dtype)
    num_windows = (series.shape[0] - window_size) // stride + 1
    return series.as_strided(
        size=(num_windows, window_size, series.shape[1]),
        stride=(series.stride(0) * stride, series.stride(0), series.stride(1)),
    ).clone()


def load_dl_set(path: str) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    required = {"feature_columns", "price_columns", "window_size", "stride", "train_series", "test_series", "train_windows"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"DL set missing keys: {sorted(missing)}")
    return data


def load_stats_set(path: str) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    required = {"feature_columns", "price_columns", "train_series", "test_series", "full_series"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Stats set missing keys: {sorted(missing)}")
    return data


def _empty_series_like(series: torch.Tensor) -> torch.Tensor:
    return torch.empty((0, series.shape[1]), dtype=series.dtype)


def _empty_windows_like(series: torch.Tensor, window_size: int) -> torch.Tensor:
    return torch.empty((0, window_size, series.shape[1]), dtype=series.dtype)


def build_batch_from_dl_set(dl_set: Dict[str, Any], generation_length: int) -> StandardBatch:
    train_series = dl_set["train_series"].float()
    test_series = dl_set["test_series"].float()
    train_windows = dl_set["train_windows"].float()
    window_size = int(dl_set["window_size"])

    valid_series = dl_set.get("valid_series", _empty_series_like(train_series)).float()
    valid_windows = dl_set.get("valid_windows", _empty_windows_like(train_series, window_size)).float()
    test_windows = dl_set.get("test_windows")
    if test_windows is None:
        test_windows = sliding_window_2d(test_series, generation_length, stride=1)
    else:
        test_windows = test_windows.float()

    channel_count = train_series.shape[1]
    empty_initial = torch.empty((0, channel_count), dtype=train_series.dtype)

    return StandardBatch(
        train=train_series,
        valid=valid_series,
        test=test_series,
        train_initial=train_series[0] if train_series.shape[0] else torch.zeros(channel_count, dtype=train_series.dtype),
        valid_initial=valid_series[0] if valid_series.shape[0] else torch.zeros(channel_count, dtype=train_series.dtype),
        test_initial=test_series[0] if test_series.shape[0] else torch.zeros(channel_count, dtype=train_series.dtype),
        asset_columns=list(dl_set["feature_columns"]),
        price_columns=list(dl_set["price_columns"]),
        train_windows=train_windows,
        valid_windows=valid_windows,
        test_windows=test_windows,
        train_window_initials=train_windows[:, 0, :] if train_windows.shape[0] else empty_initial,
        valid_window_initials=valid_windows[:, 0, :] if valid_windows.shape[0] else empty_initial,
        test_window_initials=test_windows[:, 0, :] if test_windows.shape[0] else empty_initial,
        inferred_length=window_size,
    )


def build_batch_from_stats_set(stats_set: Dict[str, Any], generation_length: int) -> StandardBatch:
    train_series = stats_set["train_series"].float()
    test_series = stats_set["test_series"].float()
    channel_count = train_series.shape[1]
    test_windows = sliding_window_2d(test_series, generation_length, stride=1)
    empty_initial = torch.empty((0, channel_count), dtype=train_series.dtype)

    return StandardBatch(
        train=train_series,
        valid=_empty_series_like(train_series),
        test=test_series,
        train_initial=train_series[0] if train_series.shape[0] else torch.zeros(channel_count, dtype=train_series.dtype),
        valid_initial=torch.zeros(channel_count, dtype=train_series.dtype),
        test_initial=test_series[0] if test_series.shape[0] else torch.zeros(channel_count, dtype=train_series.dtype),
        asset_columns=list(stats_set["feature_columns"]),
        price_columns=list(stats_set["price_columns"]),
        train_windows=_empty_windows_like(train_series, generation_length),
        valid_windows=_empty_windows_like(train_series, generation_length),
        test_windows=test_windows,
        train_window_initials=empty_initial,
        valid_window_initials=empty_initial,
        test_window_initials=test_windows[:, 0, :] if test_windows.shape[0] else empty_initial,
        inferred_length=generation_length,
    )
