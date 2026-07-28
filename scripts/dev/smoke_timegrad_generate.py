#!/usr/bin/env python3
"""Smoke validation for timegrad artifacts at outputs/results/<run_id>/timegrad/artifacts/.

ACTUAL schema (matches artifact written by src/experiments/adapters/deep_learning/timegrad_adapter.py:fit/generate):

    ckpt = {
        "data": torch.Tensor,            # shape (num_samples, seq_length, num_channels), float32 log-returns
        "metadata": {
            "asset_columns": List[str],
            "base_length": int,
            "best_epoch": int,
            "best_val_loss": float,
            "diff_steps": int,
            "generator_version": str,
            "history_length": int,
            "is_multivariate": bool,
            "lags_seq": List[int],
            "model_checkpoint_manifest": dict,   # per-channel trainer weights
            # ... other training-written fields
        },
    }

Validates: shape (3D float), finiteness (no NaN/Inf), per-channel std non-collapse.

Returns exit code:
    0 = PASS
    1 = data validation failure (shape/finiteness/collapse)
    2 = artifact missing
    3 = torch.load raised
    4 = top-level not a dict
    5 = missing required top-level keys
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REQUIRED_TOP_KEYS = {"data", "metadata"}  # paired with src/experiments/adapters/deep_learning/timegrad_adapter.py:generate()
REQUIRED_METADATA_KEYS = {
    "asset_columns",
    "base_length",
    "history_length",
    "lags_seq",
    "diff_steps",
    "model_checkpoint_manifest",
}


def _validate_manifest(manifest: Any) -> tuple[bool, str]:
    """Best-effort validation of model_checkpoint_manifest.

    Accepts shapes emitted by src/experiments/adapters/deep_learning/timegrad_adapter.py
    via src/experiments/core/io.orchesration_handler (which serializes AdapterGenerateOutput.checkpoints
    to model_checkpoint_manifest directly):
      - list[str|Path]: list of checkpoint file paths consolidated / per-channel (fresh artifacts)
      - list[dict]: per-channel state_dict entries (alternative emit)
      - dict: flat key->state_dict mapping (legacy / break-glass)

    Returns (True, info_str) on pass, (False, reason) on fail.
    """
    if manifest is None:
        return True, "info: model_checkpoint_manifest is None — likely pre-fit artifact"
    if isinstance(manifest, list):
        if len(manifest) == 0:
            return False, "model_checkpoint_manifest is an empty list"
        # Path-or-string emissions (consolidated / per-channel checkpoint locators).
        if all(isinstance(e, (str, Path)) for e in manifest):
            from os.path import isfile as _isfile
            missing = [e for e in manifest if not _isfile(str(e))]
            msg = f"list[str|Path] len={len(manifest)}"
            if missing:
                msg += f" MISSING={len(missing)}/{len(manifest)}"
            return True, msg
        # Per-channel state_dict entries.
        if all(isinstance(e, dict) for e in manifest):
            return True, f"list[dict] len={len(manifest)}"
        return False, "model_checkpoint_manifest list contains non-dict non-str entries"
    if isinstance(manifest, dict):
        if len(manifest) == 0:
            return False, "model_checkpoint_manifest is an empty dict"
        return True, f"dict len={len(manifest)}"
    return False, f"model_checkpoint_manifest is neither dict, list[str], nor list[dict] (type={type(manifest).__name__})"


def validate(artifact_path: Path) -> int:
    print(f"=== VALIDATING {artifact_path} ===")

    # Existence
    if not artifact_path.exists():
        print(f"FAIL: artifact not found at {artifact_path}")
        return 2

    # Load
    try:
        ckpt = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"FAIL: torch.load raised {type(exc).__name__}: {exc}")
        return 3

    if not isinstance(ckpt, dict):
        print(f"FAIL: top-level ckpt is not a dict (type={type(ckpt).__name__})")
        return 4

    missing_top = REQUIRED_TOP_KEYS - set(ckpt.keys())
    if missing_top:
        print(f"FAIL: missing top-level keys: {sorted(missing_top)}")
        return 5

    data = ckpt["data"]
    metadata = ckpt["metadata"]
    issues: list[str] = []

    # === data validation ===
    if not isinstance(data, torch.Tensor):
        issues.append(f"data is not Tensor (type={type(data).__name__})")
    elif data.dim() != 3:
        issues.append(f"data.dim()={data.dim()} (expected 3: [samples, seq_len, channels])")
    elif not data.is_floating_point():
        issues.append(f"data.dtype={data.dtype} (expected floating)")
    else:
        num_samples, seq_len, num_channels = data.shape
        print(f"data.shape = ({num_samples}, {seq_len}, {num_channels})")
        print(f"data.dtype = {data.dtype}")

        nan_count = int(torch.isnan(data).sum())
        inf_count = int(torch.isinf(data).sum())
        print(f"NaN count: {nan_count}")
        print(f"Inf count: {inf_count}")
        if nan_count:
            issues.append(f"{nan_count} NaN values in data")
        if inf_count:
            issues.append(f"{inf_count} Inf values in data")

        finite_mask = torch.isfinite(data)
        if finite_mask.any():
            d_fin = data[finite_mask]
            print(
                f"data (finite only) "
                f"min={float(d_fin.min()):.6f} "
                f"max={float(d_fin.max()):.6f} "
                f"mean={float(d_fin.mean()):.6f}"
            )
            print(f"data (finite only) global_std={float(d_fin.std()):.6f}")

        # Per-channel std along (samples, time) — last dim is channels.
        per_chan_std = data.std(dim=(0, 1))
        nan_in_std = int(torch.isnan(per_chan_std).sum())
        if nan_in_std:
            issues.append(f"{nan_in_std} channels have NaN std")
        if torch.isinf(per_chan_std).any():
            issues.append("some channels have Inf std")
        pc_min = float(per_chan_std.min())
        pc_mean = float(per_chan_std.mean())
        pc_max = float(per_chan_std.max())
        print(
            f"per_channel_std (n={num_channels}): "
            f"min={pc_min:.6f} mean={pc_mean:.6f} max={pc_max:.6f}"
        )

        zero_std_thresh = 1e-6
        zero_std = int((per_chan_std < zero_std_thresh).sum())
        if zero_std > 0:
            issues.append(
                f"{zero_std}/{num_channels} channels have std < {zero_std_thresh} (mode collapse)"
            )

    # === metadata ===
    missing_meta = REQUIRED_METADATA_KEYS - set(metadata.keys())
    if missing_meta:
        print(f"WARN: missing metadata keys: {sorted(missing_meta)}")
    print()
    print("=== METADATA SUMMARY ===")
    for k in (
        "base_length",
        "history_length",
        "best_val_loss",
        "best_epoch",
        "diff_steps",
        "is_multivariate",
        "generator_version",
    ):
        if k in metadata:
            print(f"  {k} = {metadata[k]!r}")
    if "lags_seq" in metadata:
        print(f"  lags_seq = {metadata['lags_seq']!r}")
    if "asset_columns" in metadata:
        ac = metadata["asset_columns"]
        if isinstance(ac, list):
            print(
                f"  asset_columns = list[{len(ac)}] "
                f"first5={ac[:5]} last3={ac[-3:]}"
            )
        else:
            print(f"  asset_columns = type={type(ac).__name__}")

    # Cross-contract sanity (WARN only, do not fail — adapter may legitimately
    # emit variants for partial-fit / smoke runs).
    if data.dim() == 3 and isinstance(metadata.get("asset_columns"), list):
        n_ch = data.shape[2]
        n_ac = len(metadata["asset_columns"])
        if n_ch != n_ac:
            print(f"WARN: num_channels={n_ch} != len(asset_columns)={n_ac}")
    if data.dim() == 3 and "base_length" in metadata:
        seq_len = data.shape[1]
        base_len = metadata["base_length"]
        if seq_len != base_len:
            print(f"WARN: data.shape[1]={seq_len} != base_length={base_len}")

    manifest_ok, manifest_msg = _validate_manifest(metadata.get("model_checkpoint_manifest"))
    if not manifest_ok:
        issues.append(manifest_msg)
    else:
        print(f"  model_checkpoint_manifest: {manifest_msg}")
        m = metadata["model_checkpoint_manifest"]
        if isinstance(m, list) and m and isinstance(m[0], dict):
            print(f"  manifest[0] keys: {sorted(m[0].keys())[:6]}")

    print()
    print("=== VERDICT ===")
    if not issues:
        print(f"PASS: {artifact_path}")
        return 0
    print(f"FAIL: {len(issues)} issue(s) with {artifact_path}")
    for i, reason in enumerate(issues, 1):
        print(f"  {i}. {reason}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke validation for timegrad artifacts.")
    parser.add_argument("--artifact", type=Path, required=True, help="Path to timegrad .pt artifact")
    args = parser.parse_args()
    sys.exit(validate(args.artifact))


if __name__ == "__main__":
    main()
