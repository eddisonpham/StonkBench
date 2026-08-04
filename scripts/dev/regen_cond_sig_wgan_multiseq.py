#!/usr/bin/env python3
"""Per-sequence-length sanity plots for cond_sig_wgan (round-2 noise10 winner).

The standard regen_sanity.py only plots seq252. This script adds the
21 / 42 / 126 / 252 comparison the user asked for, writing to
  outputs/sanity/latest/cond_sig_wgan/seq{L}/<ASSET>/{overlay,hist}.png
overwriting the stale plots left by the old collapsed model (same paths, so
no new top-level folders are introduced).

Both sims and GT artifacts are already in raw log-return scale (the pipeline
denormalizes before saving), so we plot them as-is. The overlay shows ALL GT
test windows as thin black lines so the GT variance envelope is directly
comparable to the blue simulation envelope.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS = Path("/scratch/epham/stonkbench/output/results/latest")
SANITY = Path("/scratch/epham/stonkbench/output/sanity/latest/cond_sig_wgan")
MODEL = "cond_sig_wgan"
SEQ_LENGTHS = [21, 42, 126, 252]


def load_artifact(p: Path) -> torch.Tensor:
    obj = torch.load(str(p), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"].float()
    return obj.float()


def plot_hist_pooled(
    gt_windows: torch.Tensor,
    sims: torch.Tensor,
    channel_idx: int,
    channel_name: str,
    output_path: Path,
) -> None:
    """Histogram of ALL GT windows (black) vs ALL sims (blue), pooled."""
    gt = gt_windows[:, :, channel_idx].detach().cpu().numpy().reshape(-1)
    sims_flat = sims[:, :, channel_idx].detach().cpu().numpy().reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = 60
    axes[0].hist(gt, bins=bins, color="black", alpha=0.85, density=True)
    axes[0].set_title(
        f"{channel_name}: ground truth (N={gt_windows.shape[0]} windows)"
    )
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("density")
    axes[1].hist(sims_flat, bins=bins, color="tab:blue", alpha=0.7, density=True)
    axes[1].set_title(f"{channel_name}: simulations (all)")
    axes[1].set_xlabel("value")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_overlay_all_gt(
    gt_windows: torch.Tensor,
    sims: torch.Tensor,
    channel_idx: int,
    channel_name: str,
    output_path: Path,
) -> None:
    """Overlay all GT windows (thin black) + all sims (thin blue)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    t_sim = np.arange(sims.shape[1])
    for sim_idx in range(sims.shape[0]):
        ax.plot(
            t_sim,
            sims[sim_idx, :, channel_idx].detach().cpu().numpy(),
            color="tab:blue",
            alpha=0.12,
            linewidth=0.8,
        )
    for gt_idx in range(gt_windows.shape[0]):
        ax.plot(
            np.arange(gt_windows.shape[1]),
            gt_windows[gt_idx, :, channel_idx].detach().cpu().numpy(),
            color="black",
            alpha=0.45,
            linewidth=1.2,
        )
    ax.set_title(
        f"{channel_name}: sim N={sims.shape[0]} vs GT windows N={gt_windows.shape[0]}"
    )
    ax.set_xlabel("time step")
    ax.set_ylabel("log return")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    from src.utils.preprocessed_data_utils import load_dl_set, resolve_dl_set_path

    dl_set = load_dl_set(resolve_dl_set_path())
    feature_columns = list(dl_set["feature_columns"])

    total = 0
    for L in SEQ_LENGTHS:
        sims = load_artifact(RESULTS / MODEL / "artifacts" / f"{MODEL}_seq{L}.pt")
        gt = load_artifact(RESULTS / "ground_truth" / f"ground_truth_seq{L}.pt")
        n_assets = min(sims.shape[-1], gt.shape[-1], len(feature_columns))
        out_dir = SANITY / f"seq{L}"
        for i in range(n_assets):
            name = feature_columns[i]
            plot_overlay_all_gt(
                gt, sims, i, name, out_dir / name / "overlay.png"
            )
            plot_hist_pooled(gt, sims, i, name, out_dir / name / "hist.png")
            total += 2
        print(
            f"  seq{L}: sims={tuple(sims.shape)} gt={tuple(gt.shape)} "
            f"-> {n_assets} assets in {out_dir}"
        )
    print(f"=== wrote {total} PNGs for {MODEL} ===")


if __name__ == "__main__":
    main()
