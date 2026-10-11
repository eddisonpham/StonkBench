#!/usr/bin/env python
"""Simulation-vs-GT overlay plots for manual inspection — log returns only.

Writes ``<report_dir>/simulations/<model>/seq<L>.png`` for every model and
sequence length:

  black lines  = ground-truth test windows (denormalized log returns,
                  rebuilt exactly like score_outputs/evaluator)
  blue lines   = a seeded subsample of the generated artifact samples
                 (artifacts are saved denormalized by the pipeline)

Panels: one per asset channel (grid), titled with the ticker, zero line for
reference. y-limits span the FULL range of GT + all generated samples so no
spike or outlier is hidden from inspection.

Usage:
    python scripts/mini_exp/plot_simulations.py \
        --run_id fair_20261004 \
        --output_root /home/epham/StonkBench/outputs \
        --report_dir /home/epham/StonkBench/outputs/mini_exp/fair_20261004
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.preprocessed_data_utils import (  # noqa: E402
    channel_norm_stats,
    denormalize_channels,
    load_dl_set,
    resolve_dl_set_path,
    sliding_window_2d,
)

DEFAULT_MODELS = [
    "unconditional_tsdiffusion",
    "conditional_tsdiffusion",
    "quantgan",
    "timegrad",
    "vrnn_klsched",
    "kalman_vae_safe",
    "cond_sig_wgan",
]


def load_artifact(path: Path) -> np.ndarray:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("data", "samples", "series", "tensor"):
            if key in raw and torch.is_tensor(raw[key]):
                return raw[key].float().numpy()
        raise ValueError(f"no tensor in {path}: keys={list(raw)}")
    return np.asarray(raw, dtype=np.float64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_id", default="fair_20261004")
    p.add_argument("--output_root", default="/home/epham/StonkBench/outputs")
    p.add_argument("--report_dir",
                   default="/home/epham/StonkBench/outputs/mini_exp/fair_20261004")
    p.add_argument("--seq_lengths", default="21,42,126,252")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--n_sims", type=int, default=25,
                   help="simulation paths drawn per panel (subsample of all)")
    p.add_argument("--n_gt", type=int, default=5,
                   help="GT windows drawn per panel (evenly spaced)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    models = args.models or DEFAULT_MODELS
    seq_lengths = [int(x) for x in args.seq_lengths.split(",") if x.strip()]
    results_dir = Path(args.output_root) / "results" / args.run_id
    out_root = Path(args.report_dir) / "simulations"

    dl_set = load_dl_set(resolve_dl_set_path())
    mean, std = channel_norm_stats(dl_set)
    test_raw = denormalize_channels(dl_set["test_series"].float(), mean, std)
    channels = list(dl_set["feature_columns"])
    gt_by_seq = {
        L: sliding_window_2d(test_raw, L, stride=1).numpy() for L in seq_lengths
    }

    rng = np.random.default_rng(args.seed)
    written, missing = [], []
    for model in models:
        art_dir = results_dir / model / "artifacts"
        for L in seq_lengths:
            art = art_dir / f"{model}_seq{L}.pt"
            if not art.is_file():
                missing.append(str(art))
                continue
            gen = load_artifact(art)          # (n, L, C) raw log returns
            gt = gt_by_seq[L]                  # (m, L, C) raw log returns
            if gt.shape[0] == 0:
                missing.append(f"{model} seq{L}: no GT windows")
                continue
            n_channels = min(gen.shape[-1], len(channels))

            gt_idx = np.unique(np.linspace(0, gt.shape[0] - 1,
                                           min(args.n_gt, gt.shape[0])).astype(int))
            sim_idx = rng.choice(gen.shape[0],
                                 size=min(args.n_sims, gen.shape[0]),
                                 replace=False)

            # Full data range across ALL samples + GT: nothing is clipped.
            lo = min(float(gt.min()), float(gen.min()))
            hi = max(float(gt.max()), float(gen.max()))
            pad = 0.05 * (hi - lo) if hi > lo else 1e-6

            ncols = 5
            nrows = math.ceil(n_channels / ncols)
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(4.6 * ncols, 2.7 * nrows),
                                     squeeze=False)
            for c in range(nrows * ncols):
                ax = axes[c // ncols][c % ncols]
                if c >= n_channels:
                    ax.axis("off")
                    continue
                for g in gt_idx:
                    ax.plot(gt[g, :, c], color="black", lw=1.3, alpha=0.85)
                for s in sim_idx:
                    ax.plot(gen[s, :, c], color="tab:blue", lw=0.8, alpha=0.18)
                ax.axhline(0.0, color="gray", lw=0.6, alpha=0.5)
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_title(channels[c], fontsize=9)
                ax.tick_params(labelsize=7)
                if c % ncols == 0:
                    ax.set_ylabel("log return", fontsize=8)
                if c // ncols == nrows - 1:
                    ax.set_xlabel("time step", fontsize=8)
                ax.grid(alpha=0.25)
            fig.suptitle(
                f"{model} — seq{L}: {len(sim_idx)} simulations (blue) vs "
                f"{len(gt_idx)} GT test windows (black), log returns "
                f"[n_gen={gen.shape[0]}, n_gt={gt.shape[0]}]",
                fontsize=13,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.965])
            out = out_root / model / f"seq{L}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=110)
            plt.close(fig)
            written.append(str(out))
            print(f"WROTE {out}", flush=True)

    print(f"\n{len(written)} plots under {out_root}")
    if missing:
        print("MISSING:")
        for m in missing:
            print(f"  {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
