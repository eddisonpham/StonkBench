"""Post-training sanity plots + per-channel metrics: ground truth vs simulations."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import StandardBatch
from src.utils.preprocessed_data_utils import (
    channel_norm_stats,
    denormalize_channels,
    load_dl_set,
    resolve_dl_set_path,
)

NUM_SIMULATIONS = 50


@dataclass
class ChannelMetrics:
    """Per-channel distribution-difference summary between GT and sim ensemble."""

    channel: str
    channel_idx: int
    gt_mean: float
    gt_std: float
    gt_skew: float
    gt_kurt: float
    sim_mean: float
    sim_std: float
    sim_skew: float
    sim_kurt: float
    mean_diff: float
    std_ratio: float
    skew_diff: float
    kurt_diff: float
    ks_stat: float
    ks_pvalue: float

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_channel_metrics(
    gt_channel: np.ndarray,
    sim_channels: np.ndarray,
    channel: str,
    channel_idx: int,
) -> ChannelMetrics:
    from scipy import stats as sp_stats

    gt_flat = gt_channel.reshape(-1)
    sim_flat = sim_channels.reshape(-1)

    try:
        ks_stat, ks_pval = sp_stats.ks_2samp(gt_flat, sim_flat)
    except Exception:
        ks_stat, ks_pval = float("nan"), 0.0
    if not np.isfinite(ks_stat):
        ks_stat, ks_pval = 0.0, 0.0

    gt_mean = float(gt_flat.mean())
    gt_std = float(gt_flat.std() + 1e-12)
    gt_skew = float(sp_stats.skew(gt_flat)) if gt_flat.size > 2 else 0.0
    gt_kurt = float(sp_stats.kurtosis(gt_flat)) if gt_flat.size > 3 else 0.0

    sim_mean = float(sim_flat.mean())
    sim_std = float(sim_flat.std() + 1e-12)
    sim_skew = float(sp_stats.skew(sim_flat)) if sim_flat.size > 2 else 0.0
    sim_kurt = float(sp_stats.kurtosis(sim_flat)) if sim_flat.size > 3 else 0.0

    return ChannelMetrics(
        channel=channel,
        channel_idx=channel_idx,
        gt_mean=gt_mean,
        gt_std=gt_std,
        gt_skew=gt_skew,
        gt_kurt=gt_kurt,
        sim_mean=sim_mean,
        sim_std=sim_std,
        sim_skew=sim_skew,
        sim_kurt=sim_kurt,
        mean_diff=sim_mean - gt_mean,
        std_ratio=sim_std / gt_std,
        skew_diff=sim_skew - gt_skew,
        kurt_diff=sim_kurt - gt_kurt,
        ks_stat=float(ks_stat),
        ks_pvalue=float(ks_pval),
    )


def _plot_overlay(
    ground_truth: torch.Tensor,
    simulations: torch.Tensor,
    channel_idx: int,
    channel_name: str,
    output_path: Path,
) -> None:
    gt = ground_truth[:, channel_idx].detach().cpu().numpy()
    sims = simulations[:, :, channel_idx].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    t_sim = np.arange(sims.shape[1])
    t_gt = np.arange(gt.shape[0])
    for sim_idx in range(sims.shape[0]):
        ax.plot(t_sim, sims[sim_idx], color="tab:blue", alpha=0.12, linewidth=0.8)
    ax.plot(
        t_gt,
        gt,
        color="black",
        linewidth=2.0,
        label=f"ground truth (test window)\nsim N={sims.shape[0]}",
    )
    ax.set_title(f"{channel_name}: ground truth vs {sims.shape[0]} simulations")
    ax.set_xlabel("time step")
    ax.set_ylabel("log return")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _plot_histogram(
    ground_truth: torch.Tensor,
    simulations: torch.Tensor,
    channel_idx: int,
    channel_name: str,
    output_path: Path,
) -> None:
    gt = ground_truth[:, channel_idx].detach().cpu().numpy().reshape(-1)
    sims = simulations[:, :, channel_idx].detach().cpu().numpy().reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = 60
    axes[0].hist(gt, bins=bins, color="black", alpha=0.85, density=True)
    axes[0].set_title(f"{channel_name}: ground truth")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("density")
    axes[1].hist(sims, bins=bins, color="tab:blue", alpha=0.7, density=True)
    axes[1].set_title(f"{channel_name}: simulations (all)")
    axes[1].set_xlabel("value")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _write_per_channel_csv(metrics: List[ChannelMetrics], output_path: Path) -> None:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ChannelMetrics.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in metrics:
            writer.writerow(m.to_row())


def _write_per_channel_json(metrics: List[ChannelMetrics], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "channel_count": len(metrics),
        "channels": [m.to_row() for m in metrics],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def render_model_sanity(
    adapter: ModelAdapter,
    batch: StandardBatch,
    output_dir: Path,
    generation_length: int,
    seed: int,
    feature_columns: Sequence[str] | None = None,
) -> List[Path]:
    """Render per-channel sanity plots + CSV/JSON summary."""
    if batch.valid_windows is None or batch.valid_windows.shape[0] == 0:
        raise ValueError("Sanity visualization requires at least one validation window.")

    dl_set = load_dl_set(resolve_dl_set_path())
    feature_columns = feature_columns or dl_set["feature_columns"]
    stats = channel_norm_stats(dl_set)

    n_channels_observed = int(dl_set["train_series"].shape[1])
    if len(feature_columns) != n_channels_observed:
        feature_columns = dl_set["feature_columns"]

    ground_truth = batch.valid_windows[0].float()
    if stats is not None:
        ground_truth = denormalize_channels(ground_truth.unsqueeze(0), *stats).squeeze(0)

    generated = adapter.generate(
        num_samples=NUM_SIMULATIONS,
        generation_length=generation_length,
        seed=seed,
    )
    simulations = generated.data.float()
    if stats is not None:
        simulations = denormalize_channels(simulations, *stats)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    metrics: List[ChannelMetrics] = []

    for idx, name in enumerate(feature_columns):
        if idx >= ground_truth.shape[1] or idx >= simulations.shape[2]:
            continue
        channel_dir = output_dir / name
        channel_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = channel_dir / "overlay.png"
        hist_path = channel_dir / "hist.png"

        _plot_overlay(ground_truth, simulations, idx, name, overlay_path)
        _plot_histogram(ground_truth, simulations, idx, name, hist_path)
        saved.extend([overlay_path, hist_path])

        gt_1d = ground_truth[:, idx].detach().cpu().numpy()
        sim_2d = simulations[:, :, idx].detach().cpu().numpy()
        metrics.append(_compute_channel_metrics(gt_1d, sim_2d, name, idx))

    csv_path = output_dir / "per_channel_summary.csv"
    json_path = output_dir / "per_channel_summary.json"
    _write_per_channel_csv(metrics, csv_path)
    _write_per_channel_json(metrics, json_path)
    saved.extend([csv_path, json_path])

    manifest = {
        "model_name": getattr(adapter, "model_name", "unknown"),
        "generation_length": int(generation_length),
        "num_simulations": int(NUM_SIMULATIONS),
        "channel_count": len(metrics),
        "channels": [m.channel for m in metrics],
        "files": sorted({str(p) for p in saved}),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    saved.append(manifest_path)
    return saved
