#!/usr/bin/env python3
"""Reconstruct the exact real-data reference each evaluated cell was scored against.

The evaluator (``src/unified_evaluator.py``) does **not** score against
``outputs/ground_truth/*.pt``.  For every target length ``L`` it builds, from
``data/preprocessed/dl_set_W252.pt``:

    all_windows = sliding_window_2d(dl_set["test_series_raw"], L, stride=1)   # T-L+1 x L x 25
    split       = len(all_windows) // 2
    statistical reference = all_windows                       (then truncated to num_samples)
    deep-learning reference = all_windows[split:]             (then truncated to num_samples)

and then aligns both sides with ``n_use = min(len(reference), num_samples)``.

This script rebuilds those references, proves the reconstruction against the
shipped metrics, and writes the verified arrays to
``handover/03_ground_truth/reference/<model>_seq<L>.npz`` plus the alignment
table at ``scripts/handover/data/evaluation_alignment.csv``.

Verification: ``StylizedFactsEvaluator.autocorr_returns.{real,synth}`` and
``volatility_clustering.{real,synth}`` are recomputed from the reconstruction and
compared with the metrics recorded in
``handover/04_evaluation_results/complete_evaluation.json`` for all 48 cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from torch_pt_reader import load_pt

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parents[1]
HANDOVER = REPO / "handover"

DL_SET = REPO / "data" / "preprocessed" / "dl_set_W252.pt"
EVAL = HANDOVER / "04_evaluation_results"
REF_DIR = HANDOVER / "03_ground_truth" / "reference"
SEQ_LENGTHS = (21, 42, 126, 252)


def sliding_window_2d(series: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """Torch-free equivalent of ``src/utils/preprocessed_data_utils.sliding_window_2d``."""
    T = series.shape[0]
    n = (T - window) // stride + 1
    if n <= 0:
        return np.empty((0, window, series.shape[1]), dtype=series.dtype)
    idx = np.arange(n)[:, None] * stride + np.arange(window)[None, :]
    return series[idx]


def lag1_autocorr_mean(data: np.ndarray) -> float:
    """``src/taxonomies/stylized_facts.autocorr_returns`` averaged over channels."""
    arr = np.asarray(data, dtype=np.float64)
    per_channel = []
    for c in range(arr.shape[2]):
        acfs = []
        for sample in arr[:, :, c]:
            m = sample.mean()
            den = np.sum((sample - m) ** 2)
            if den != 0:
                acfs.append(np.sum((sample[:-1] - m) * (sample[1:] - m)) / den)
        if acfs:
            per_channel.append(np.mean(acfs))
    return float(np.mean(per_channel))


def vol_clustering_mean(data: np.ndarray) -> float:
    """``volatility_clustering`` at lag 1, averaged over channels."""
    arr = np.asarray(data, dtype=np.float64)
    per_channel = []
    for c in range(arr.shape[2]):
        vals = []
        for sample in arr[:, :, c]:
            sq = sample ** 2
            m = sq.mean()
            den = np.sum((sq - m) ** 2)
            if den != 0:
                vals.append(np.sum((sq[:-1] - m) * (sq[1:] - m)) / den)
        if vals:
            per_channel.append(np.mean(vals))
    return float(np.mean(per_channel))


def build_references() -> dict[int, np.ndarray]:
    dl = load_pt(DL_SET)
    series = np.asarray(dl["test_series_raw"], dtype=np.float32)
    return {L: sliding_window_2d(series, L) for L in SEQ_LENGTHS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true", help="verify only, do not write the reference arrays")
    args = ap.parse_args()

    windows = build_references()
    for L, w in windows.items():
        print(f"  windows over dl_set_W252 test_series_raw at L={L:>3}: {w.shape}")

    rows, failures = [], []
    worst = 0.0
    bundle = json.loads((EVAL / "complete_evaluation.json").read_text())
    for L in SEQ_LENGTHS:
        all_windows = windows[L]
        split = all_windows.shape[0] // 2
        for cell in bundle.values():
            if int(cell["evaluated_at_length"]) != L:
                continue
            model, mtype = cell["model_name"], cell["model_type"]
            n_gen = int(cell["num_samples"])
            reference = all_windows if mtype == "statistical" else all_windows[split:]
            n_use = min(reference.shape[0], n_gen)

            real = reference[:n_use]
            gen_all = load_pt(HANDOVER / "02_generated_outputs" / model / f"{model}_seq{L}.pt")["data"]
            gen = np.asarray(gen_all[:n_use], dtype=np.float32)

            sf = cell.get("StylizedFactsEvaluator", {})
            checks = {
                "autocorr_returns.real": (lag1_autocorr_mean(real), sf.get("autocorr_returns", {}).get("real")),
                "autocorr_returns.synth": (lag1_autocorr_mean(gen), sf.get("autocorr_returns", {}).get("synth")),
                "volatility_clustering.real": (vol_clustering_mean(real), sf.get("volatility_clustering", {}).get("real")),
                "volatility_clustering.synth": (vol_clustering_mean(gen), sf.get("volatility_clustering", {}).get("synth")),
            }
            row = {
                "cell": f"{model}_seq{L}",
                "model": model,
                "model_type": mtype,
                "sequence_length": L,
                "num_samples_shipped": n_gen,
                "reference_pool": "all_windows" if mtype == "statistical" else "second_half",
                "reference_pool_size": reference.shape[0],
                "n_used_by_evaluator": n_use,
            }
            for name, (got, want) in checks.items():
                if want is None:
                    row[name] = ""
                    continue
                d = abs(got - want)
                worst = max(worst, d)
                row[name] = round(d, 12)
                if d > 1e-6:
                    failures.append((f"{model}_seq{L}", name, got, want))
            rows.append(row)

    print(f"\nreconstruction check over {len(rows)} cells x 4 values: worst |delta| = {worst:.3e}")
    if failures:
        print(f"FAIL: {len(failures)} mismatches")
        for f in failures[:10]:
            print("   ", f)
    else:
        print("OK: every reconstructed reference reproduces the shipped stylised-fact metrics")

    alignment = CODE_DIR / "data" / "evaluation_alignment.csv"
    alignment.parent.mkdir(parents=True, exist_ok=True)
    with alignment.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {alignment.relative_to(REPO)}")

    if not args.no_write:
        REF_DIR.mkdir(parents=True, exist_ok=True)
        for L in SEQ_LENGTHS:
            all_windows = windows[L]
            split = all_windows.shape[0] // 2
            save(all_windows, REF_DIR / f"reference_statistical_seq{L}.pt", L)
            save(all_windows[split:], REF_DIR / f"reference_deep_learning_seq{L}.pt", L)
        for row in rows:
            L = row["sequence_length"]
            all_windows = windows[L]
            split = all_windows.shape[0] // 2
            pool = all_windows if row["model_type"] == "statistical" else all_windows[split:]
            save(pool[: row["n_used_by_evaluator"]], REF_DIR / f"{row['cell']}.pt", L)
        print(f"wrote per-cell verified references to {REF_DIR.relative_to(REPO)}")

    if failures:
        sys.exit(1)


def save(array: np.ndarray, path: Path, L: int) -> None:
    """Write as .npz (numpy, no torch needed) so the arrays are trivially readable."""
    path = path.with_suffix(".npz")
    np.savez(
        path,
        data=array.astype(np.float32),
        sequence_length=np.int64(L),
        num_samples=np.int64(array.shape[0]),
        source=np.array(
            "reconstructed from data/preprocessed/dl_set_W252.pt test_series_raw via stride-1 windows"
        ),
    )


if __name__ == "__main__":
    main()
