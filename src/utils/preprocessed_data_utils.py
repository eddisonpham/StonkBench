"""Helpers for loading model-ready preprocessed tensors."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from src.experiments.core.contracts import StandardBatch


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


def _empty_series_like(series: torch.Tensor) -> torch.Tensor:
    return torch.empty((0, series.shape[1]), dtype=series.dtype)


def _empty_windows_like(series: torch.Tensor, window_size: int) -> torch.Tensor:
    return torch.empty((0, window_size, series.shape[1]), dtype=series.dtype)


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


def build_batch_from_dl_set(dl_set: Dict[str, Any], generation_length: int) -> StandardBatch:
    train_series = dl_set["train_series"].float()
    test_series = dl_set["test_series"].float()
    train_windows = dl_set["train_windows"].float()
    test_windows = sliding_window_2d(test_series, generation_length, stride=1)

    # Utility evaluators expect non-empty valid/test splits; split test windows in half.
    split_idx = test_windows.shape[0] // 2
    valid_windows = test_windows[:split_idx]
    eval_test_windows = test_windows[split_idx:]
    if eval_test_windows.shape[0] == 0:
        eval_test_windows = valid_windows

    valid_series = _empty_series_like(train_series)
    eval_test_series = test_series

    channel_count = train_series.shape[1]
    empty_initial = torch.empty((0, channel_count), dtype=train_series.dtype)
    train_window_initials = train_windows[:, 0, :] if train_windows.shape[0] else empty_initial
    valid_window_initials = valid_windows[:, 0, :] if valid_windows.shape[0] else empty_initial
    test_window_initials = eval_test_windows[:, 0, :] if eval_test_windows.shape[0] else empty_initial
    train_initial = train_series[0] if train_series.shape[0] else torch.zeros(channel_count, dtype=train_series.dtype)

    return StandardBatch(
        train=train_series,
        valid=valid_series,
        test=eval_test_series,
        train_initial=train_initial,
        valid_initial=torch.zeros(channel_count, dtype=train_series.dtype),
        test_initial=eval_test_series[0] if eval_test_series.shape[0] else torch.zeros(channel_count, dtype=train_series.dtype),
        asset_columns=list(dl_set["feature_columns"]),
        price_columns=list(dl_set["price_columns"]),
        train_windows=train_windows,
        valid_windows=valid_windows if valid_windows.shape[0] else _empty_windows_like(train_series, generation_length),
        test_windows=eval_test_windows if eval_test_windows.shape[0] else _empty_windows_like(train_series, generation_length),
        train_window_initials=train_window_initials,
        valid_window_initials=valid_window_initials,
        test_window_initials=test_window_initials,
        inferred_length=int(dl_set["window_size"]),
    )


def build_batch_from_stats_set(stats_set: Dict[str, Any], generation_length: int) -> StandardBatch:
    train_series = stats_set["train_series"].float()
    test_series = stats_set["test_series"].float()
    channel_count = train_series.shape[1]
    test_windows = sliding_window_2d(test_series, generation_length, stride=1)
    empty_initial = torch.empty((0, channel_count), dtype=train_series.dtype)
    train_initial = train_series[0] if train_series.shape[0] else torch.zeros(channel_count, dtype=train_series.dtype)

    return StandardBatch(
        train=train_series,
        valid=_empty_series_like(train_series),
        test=test_series,
        train_initial=train_initial,
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
