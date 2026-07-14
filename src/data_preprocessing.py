"""Preprocess combined_data.csv into model-ready tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dl_set.pt and statsmodel_set.pt from combined_data.csv."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/combined_data.csv",
        help="Path to combined_data.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/preprocessed",
        help="Directory for preprocessed .pt files",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="DL sliding window size (matches paper generation length)",
    )
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation fraction carved from the train region (before test gap)",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=None,
        help="Temporal gap between splits (default: window_size - 1)",
    )
    return parser.parse_args()


def _split_feature_columns(columns: List[str]) -> Tuple[List[str], List[str]]:
    price_cols = [c for c in columns if c != "timestamp" and not c.endswith("_volume")]
    volume_cols = [c for c in columns if c.endswith("_volume")]
    if not price_cols or not volume_cols:
        raise ValueError("Expected both price columns and *_volume columns in CSV.")
    return price_cols, volume_cols


def _sliding_windows(series: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    if series.ndim != 2:
        raise ValueError(f"Expected 2D tensor (T, C), got {tuple(series.shape)}")
    if series.shape[0] < window_size:
        return torch.empty((0, window_size, series.shape[1]), dtype=series.dtype)
    num_windows = (series.shape[0] - window_size) // stride + 1
    return series.as_strided(
        size=(num_windows, window_size, series.shape[1]),
        stride=(series.stride(0) * stride, series.stride(0), series.stride(1)),
    ).clone()


def _build_transformed_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    price_cols, volume_cols = _split_feature_columns(list(df.columns))
    for col in price_cols + volume_cols:
        if (df[col] <= 0).any():
            raise ValueError(f"Column '{col}' has non-positive values; log transform is undefined.")

    price_log_returns = np.log(df[price_cols]).diff().iloc[1:].reset_index(drop=True)
    # Log-volume changes (not levels) so volume features are stationary like returns.
    volume_log_changes = np.log(df[volume_cols]).diff().iloc[1:].reset_index(drop=True)
    timestamps = df["timestamp"].iloc[1:].reset_index(drop=True)

    transformed = pd.concat([timestamps, price_log_returns, volume_log_changes], axis=1)
    feature_columns = price_cols + volume_cols
    return transformed, price_cols, feature_columns


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    transformed_df, price_columns, feature_columns = _build_transformed_frame(df)
    features = torch.tensor(transformed_df[feature_columns].values, dtype=torch.float32)
    timestamps = torch.tensor(transformed_df["timestamp"].values, dtype=torch.long)

    window_size = int(args.window_size)
    stride = int(args.stride)
    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train_ratio must be in (0, 1)")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val_ratio must be in (0, 1)")
    gap = int(args.gap) if args.gap is not None else max(window_size - 1, 0)

    total_steps = features.shape[0]
    train_region_end = int(total_steps * train_ratio)
    test_start = min(train_region_end + gap, total_steps)

    val_len = max(window_size, int(train_region_end * val_ratio))
    val_start = max(window_size, train_region_end - val_len)
    train_fit_end = max(window_size, val_start - gap)

    train_fit_series = features[:train_fit_end]
    valid_series = features[val_start:train_region_end]
    test_series = features[test_start:]
    train_fit_timestamps = timestamps[:train_fit_end]
    valid_timestamps = timestamps[val_start:train_region_end]
    test_timestamps = timestamps[test_start:]

    # Per-channel z-score from train-fit split only (leak-free).
    channel_mean = train_fit_series.mean(dim=0)
    channel_std = train_fit_series.std(dim=0, unbiased=True).clamp(min=1e-8)
    train_series_norm = (train_fit_series - channel_mean) / channel_std
    valid_series_norm = (valid_series - channel_mean) / channel_std
    test_series_norm = (test_series - channel_mean) / channel_std

    dl_train_windows = _sliding_windows(train_series_norm, window_size, stride)
    dl_valid_windows = _sliding_windows(valid_series_norm, window_size, stride)
    dl_test_windows = _sliding_windows(test_series_norm, window_size, stride)

    dl_set: Dict[str, object] = {
        "feature_columns": feature_columns,
        "price_columns": price_columns,
        "window_size": window_size,
        "stride": stride,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "gap": gap,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "train_series": train_series_norm,
        "valid_series": valid_series_norm,
        "test_series": test_series_norm,
        "train_series_raw": train_fit_series,
        "valid_series_raw": valid_series,
        "test_series_raw": test_series,
        "train_timestamps": train_fit_timestamps,
        "valid_timestamps": valid_timestamps,
        "test_timestamps": test_timestamps,
        "train_windows": dl_train_windows,
        "valid_windows": dl_valid_windows,
        "test_windows": dl_test_windows,
    }

    statsmodel_set: Dict[str, object] = {
        "feature_columns": feature_columns,
        "price_columns": price_columns,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "gap": gap,
        "full_series": features,
        "full_timestamps": timestamps,
        "train_series": train_fit_series,
        "valid_series": valid_series,
        "test_series": test_series,
        "train_timestamps": train_fit_timestamps,
        "valid_timestamps": valid_timestamps,
        "test_timestamps": test_timestamps,
    }

    dl_path = output_dir / "dl_set.pt"
    stats_path = output_dir / "statsmodel_set.pt"
    torch.save(dl_set, dl_path)
    torch.save(statsmodel_set, stats_path)

    print(f"Saved DL set: {dl_path}")
    print(f"  train_series (norm): {tuple(train_series_norm.shape)}, train_windows: {tuple(dl_train_windows.shape)}")
    print(f"  valid_series (norm): {tuple(valid_series_norm.shape)}, valid_windows: {tuple(dl_valid_windows.shape)}")
    print(f"  test_series (norm):  {tuple(test_series_norm.shape)},  test_windows:  {tuple(dl_test_windows.shape)}")
    print(f"Saved stats set: {stats_path}")
    print(
        f"  full_series: {tuple(features.shape)}, train: {tuple(train_fit_series.shape)}, "
        f"valid: {tuple(valid_series.shape)}, test: {tuple(test_series.shape)}"
    )


if __name__ == "__main__":
    main()
