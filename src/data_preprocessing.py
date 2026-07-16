"""Preprocess combined_data.csv into model-ready tensors.

Split strategy
==============

Chronological split FIRST, sliding-window SECOND. This guarantees that
windows cannot leak across split boundaries:

  [   train_fit   ] [gap] [   val   ] [gap] [   test   ]
   0 ......... 0.7*N       0.7*N+g ... 0.8*N       0.8*N+2g ... N

Defaults: 70% / 10% / 20% split with ``gap = window_size - 1`` between
adjacent regions so the LAST window of ``train_fit`` is fully disjoint
from the FIRST window of ``val``, and the LAST window of ``val`` is fully
disjoint from the FIRST window of ``test``.

  * ``train_fit = features[0 : int(0.7*N)]``
  * ``val       = features[int(0.7*N) + gap : int(0.8*N) + gap]``
  * ``test      = features[int(0.8*N) + 2*gap : N]``

The channel-mean/std are computed from ``train_fit`` only (leak-free),
then applied to ``val`` / ``test``.

Output artifacts
================

* ``dl_set.pt``            — z-scored multivariate series + sliding windows
                              inside each region (used by every DL adapter).
* ``statsmodel_set.pt``     — raw (un-z-scored) multivariate series split
                              into train/val/test regions (used by the
                              statistical adapters which do their own fit).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# Chronological 70/10/20 split defaults. The ``--train_ratio`` / ``--val_ratio``
# CLI flags now mean "fraction of TOTAL feature length". ``--gap`` defaults to
# ``window_size - 1`` so windows cannot leak across boundaries.
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.1
DEFAULT_WINDOW_SIZE = 252
DEFAULT_STRIDE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dl_set.pt and statsmodel_set.pt from combined_data.csv "
        "(chronological 70/10/20 split, then sliding windows inside each region)."
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
    # Defaults are tuned to the canonical StonkBench setup: 1 trading year of
    # daily bars (≈252) per window, around one full year of train + 36 d of
    # val + 1 yr of test.
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=(
            "DL sliding window size L. Splits are computed BEFORE windowing and "
            "the inter-region gap defaults to L-1 so windows from adjacent "
            "regions cannot overlap."
        ),
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Sliding window stride")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Fraction of TOTAL feature length used as train_fit (default 0.7).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Fraction of TOTAL feature length used as val (default 0.1).",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=None,
        help=(
            "Temporal gap between adjacent regions. With the default None, gap = "
            "window_size - 1 which guarantees the last window of train_fit does "
            "not overlap the first window of val, and the same for val/test."
        ),
    )
    parser.add_argument(
        "--check_leakage",
        action="store_true",
        help=(
            "Sanity check: after computing regions, assert the first window of "
            "each region does not overlap the last window of the previous region."
        ),
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


def _split_boundaries(
    n_total: int,
    train_ratio: float,
    val_ratio: float,
    gap: int,
) -> Tuple[int, int, int, int]:
    """Return (train_end, val_start, val_end, test_start) absolute indices.

    All four indices are ABSOLUTE positions in the original series. The val and
    test regions follow their respective gap so windows cannot overlap across
    the boundary.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train_ratio must be in (0, 1)")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val_ratio must be in (0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"--train_ratio ({train_ratio}) + --val_ratio ({val_ratio}) must be < 1.0"
        )
    if gap < 0:
        raise ValueError("--gap must be >= 0")

    train_end = int(n_total * train_ratio)
    val_start = train_end + gap
    val_end = val_start + int(n_total * val_ratio)
    test_start = val_end + gap

    # Region length sanity (must hold for sliding-window slide to produce at
    # least one window). With default args (L=252, gap=251) and N > ~1500 this
    # is always satisfied; if not we surface a helpful error.
    return train_end, val_start, val_end, test_start


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
    user_gap = args.gap is not None
    gap = int(args.gap) if user_gap else max(window_size - 1, 0)

    n_total = int(features.shape[0])

    # Auto-fit relaxation (only when the user did not pass --gap explicitly):
    # if a strict gap = window_size - 1 would leave the test region too small
    # for even one sliding window, shrink gap to the largest value such that
    # EACH of train/val/test can still host at least one window. Chronology
    # (train < val < test) and the "no future-data leakage" property (training
    # windows never include val/test rows) are preserved at all times. We
    # only relax the *temporal adjacency* of windows across split boundaries.
    relaxed_gap_msg = None
    if not user_gap and n_total > 0:
        test_frac = max(0.0, 1.0 - train_ratio - val_ratio)
        test_available = int(n_total * test_frac)
        # We need train_len, val_len, test_len >= window_size; the gaps total
        # to 2*gap, so solve for the largest gap that keeps regions wide enough.
        max_gap = max(0, (test_available - window_size) // 2)
        if gap > max_gap:
            relaxed_gap_msg = (
                f"Strict gap (= window_size - 1 = {window_size - 1}) would leave the "
                f"test region smaller than window_size ({window_size}). "
                f"Auto-fitting gap to {max_gap} (largest value such that all three "
                f"regions still contain at least one window; chronology preserved)."
            )
            gap = max_gap

    train_end, val_start, val_end, test_start = _split_boundaries(
        n_total=n_total,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        gap=gap,
    )

    # Refuse splits that would leave any region without enough samples for one
    # sliding window.
    train_len = train_end
    val_len = val_end - val_start
    test_len = n_total - test_start
    for region_name, region_len in (("train_fit", train_len), ("val", val_len), ("test", test_len)):
        if region_len < window_size:
            raise ValueError(
                f"Region '{region_name}' has {region_len} samples but window_size={window_size}. "
                f"Either reduce --window_size, lower --train_ratio / --val_ratio, or use a longer CSV."
            )
    if val_start >= val_end:
        raise ValueError(f"val_start={val_start} >= val_end={val_end}; check --gap and --train_ratio.")
    if test_start > n_total:
        raise ValueError(
            f"test_start={test_start} exceeds n_total={n_total}; the 70/10/20 + 2*gap "
            f"payload is too large for the available feature rows."
        )

    # ===== SPLIT FIRST (chronological, with gaps) =====
    train_fit_series = features[:train_end]
    valid_series = features[val_start:val_end]
    test_series = features[test_start:]
    train_fit_timestamps = timestamps[:train_end]
    valid_timestamps = timestamps[val_start:val_end]
    test_timestamps = timestamps[test_start:]

    assert train_fit_series.shape[0] == train_end
    assert valid_series.shape[0] == val_len
    assert test_series.shape[0] == test_len

    # ===== THEN SLIDING-WINDOW INSIDE EACH REGION =====
    # Per-channel z-score from train_fit split only (leak-free).
    channel_mean = train_fit_series.mean(dim=0)
    channel_std = train_fit_series.std(dim=0, unbiased=True).clamp(min=1e-8)
    train_series_norm = (train_fit_series - channel_mean) / channel_std
    valid_series_norm = (valid_series - channel_mean) / channel_std
    test_series_norm = (test_series - channel_mean) / channel_std

    dl_train_windows = _sliding_windows(train_series_norm, window_size, stride)
    dl_valid_windows = _sliding_windows(valid_series_norm, window_size, stride)
    dl_test_windows = _sliding_windows(test_series_norm, window_size, stride)

    # Optional leakage audit: assert adjacent windows cannot overlap across
    # region boundaries. The gap >= window_size-1 makes this hold by
    # construction, but a runtime check protects against future maintenance
    # drift.
    if args.check_leakage and dl_train_windows.shape[0] and dl_valid_windows.shape[0]:
        last_train_ts = train_fit_timestamps[-1].item()
        first_val_ts = valid_timestamps[0].item()
        if int(first_val_ts) - int(last_train_ts) < int(window_size):
            raise AssertionError(
                f"Train/val leakage: last train timestamp={last_train_ts}, "
                f"first val timestamp={first_val_ts}, diff < window_size={window_size}."
            )

    dl_set: Dict[str, object] = {
        "feature_columns": feature_columns,
        "price_columns": price_columns,
        "window_size": window_size,
        "stride": stride,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "gap": gap,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
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
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
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

    if relaxed_gap_msg:
        print(f"[preprocess] NOTE: {relaxed_gap_msg}")

    print(f"Saved DL set: {dl_path}")
    print(
        f"  window_size={window_size} stride={stride} gap={gap} "
        f"split=(train={train_ratio}, val={val_ratio}, test={1 - train_ratio - val_ratio:.3f}) of N={n_total}"
    )
    print(
        f"  train_fit: rows [0, {train_end}) ({train_end} samples); "
        f"val: rows [{val_start}, {val_end}) ({val_len} samples); "
        f"test: rows [{test_start}, {n_total}) ({test_len} samples)"
    )
    print(f"  train_windows={tuple(dl_train_windows.shape)}  valid_windows={tuple(dl_valid_windows.shape)}  test_windows={tuple(dl_test_windows.shape)}")
    print(f"Saved stats set: {stats_path}")
    print(
        f"  full_series: {tuple(features.shape)}, "
        f"train={tuple(train_fit_series.shape)}, valid={tuple(valid_series.shape)}, test={tuple(test_series.shape)}"
    )


if __name__ == "__main__":
    main()
