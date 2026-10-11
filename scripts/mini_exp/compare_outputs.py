#!/usr/bin/env python
"""Manual side-by-side: fair_20261004 artifacts vs GT windows.

Hands-on checks beyond the scorer:
  - distribution shape: mean/std/kurt/lag1 of gen vs GT (flat = one number)
  - tails: 1%/99% quantiles and max|value| gen vs GT
  - degeneracy: duplicate sample rows, constant channels, per-sample std
    (does an INDIVIDUAL generated window have realistic variance?)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.utils.preprocessed_data_utils import (  # noqa: E402
    channel_norm_stats,
    denormalize_channels,
    load_dl_set,
    resolve_dl_set_path,
    sliding_window_2d,
)

SEQ = 252
RESULTS = Path("/home/epham/StonkBench/outputs/results/fair_20261004")
MODELS = [
    "unconditional_tsdiffusion",
    "conditional_tsdiffusion",
    "quantgan",
    "timegrad",
    "vrnn_klsched",
    "kalman_vae_safe",
    "cond_sig_wgan",
]


def flat_kurt(x: np.ndarray) -> float:
    c = x - x.mean(axis=0)
    s2 = (c ** 2).mean(axis=0)
    s4 = (c ** 4).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(s2 > 1e-18, s4 / (s2 ** 2) - 3.0, 0.0)
    return float(np.nanmean(k))


def lag1(x: np.ndarray) -> float:
    a, b = x[:-1], x[1:]
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt((a ** 2).mean(axis=0) * (b ** 2).mean(axis=0))
        r = np.where(denom > 1e-18, (a * b).mean(axis=0) / denom, 0.0)
    return float(np.nanmean(r))


def load_art(p: Path) -> np.ndarray:
    raw = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for k in ("data", "samples", "series", "tensor"):
            if k in raw and torch.is_tensor(raw[k]):
                return raw[k].float().numpy()
        raise SystemExit(f"no tensor in {p}: keys={list(raw)}")
    return np.asarray(raw, dtype=np.float64)


def main() -> int:
    dl_set = load_dl_set(resolve_dl_set_path())
    mean, std = channel_norm_stats(dl_set)
    test_raw = denormalize_channels(dl_set["test_series"].float(), mean, std)
    gt = sliding_window_2d(test_raw, SEQ, stride=1).numpy()
    gf = gt.reshape(-1, gt.shape[-1])

    print(f"GT seq{SEQ}: windows={gt.shape[0]}  mean={gf.mean():.5f} "
          f"std={gf.std():.4f} kurt={flat_kurt(gf):.2f} lag1={lag1(gf):.4f} "
          f"q01={np.quantile(gf,0.01):.4f} q99={np.quantile(gf,0.99):.4f} "
          f"max|v|={np.abs(gf).max():.3f} zero_frac={(gf==0).mean():.4f}")
    # individual-window std distribution of GT
    gw_std = gt.reshape(gt.shape[0], -1).std(axis=1)
    print(f"GT per-window std: q05={np.quantile(gw_std,0.05):.4f} "
          f"median={np.median(gw_std):.4f} q95={np.quantile(gw_std,0.95):.4f}")

    print(f"\n{'model':26s} {'n':>4s} {'mean':>8s} {'std':>6s} {'kurt':>6s} "
          f"{'lag1':>6s} {'q01':>7s} {'q99':>7s} {'max|v|':>7s} "
          f"{'dup%':>5s} {'constCh':>7s} {'winstd':>17s} {'nonfin':>6s}")
    for m in MODELS:
        p = RESULTS / m / "artifacts" / f"{m}_seq{SEQ}.pt"
        if not p.exists():
            print(f"{m:26s} MISSING {p}")
            continue
        gen = load_art(p).astype(np.float64)
        flat = gen.reshape(-1, gen.shape[-1])
        # duplicate rows = repeated identical samples (mode collapse signal)
        uniq = np.unique(np.round(flat, 10), axis=0).shape[0]
        dup_pct = 100.0 * (1.0 - uniq / flat.shape[0])
        const_ch = int((flat.std(axis=0) < 1e-8).sum())
        wstd = gen.reshape(gen.shape[0], -1).std(axis=1)
        nonfin = float(1.0 - np.isfinite(flat).mean())
        print(
            f"{m:26s} {flat.shape[0]:4d} {flat.mean():8.5f} {flat.std():6.4f} "
            f"{flat_kurt(flat):6.2f} {lag1(flat):6.4f} "
            f"{np.quantile(flat,0.01):7.4f} {np.quantile(flat,0.99):7.4f} "
            f"{np.abs(flat).max():7.3f} {dup_pct:5.1f} {const_ch:7d} "
            f"[{np.quantile(wstd,0.05):.3f}/{np.median(wstd):.3f}"
            f"/{np.quantile(wstd,0.95):.3f}] {nonfin:6.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
