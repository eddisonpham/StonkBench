"""Post-training sanity plots: ground truth vs many simulations per feature channel."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import StandardBatch
from src.utils.preprocessed_data_utils import channel_norm_stats, denormalize_channels, load_dl_set, resolve_dl_set_path

DEFAULT_PRICE_ASSETS = ("SPY", "AAPL")
NUM_SIMULATIONS = 50


def _feature_index(feature_columns: Sequence[str], name: str) -> int:
    try:
        return list(feature_columns).index(name)
    except ValueError as exc:
        raise KeyError(f"Feature '{name}' not found in feature_columns") from exc


def _plot_channels(
    ground_truth: torch.Tensor,
    simulations: torch.Tensor,
    channel_idx: int,
    title: str,
    output_path: Path,
) -> None:
    gt = ground_truth[:, channel_idx].detach().cpu().numpy()
    sims = simulations[:, :, channel_idx].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    time_idx = range(gt.shape[0])
    for sim_idx in range(sims.shape[0]):
        ax.plot(time_idx, sims[sim_idx], color="tab:blue", alpha=0.12, linewidth=0.8)
    ax.plot(time_idx, gt, color="black", linewidth=2.0, label="ground truth")
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("log return / log-volume change")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def channel_specs(price_assets: Sequence[str], feature_columns: Sequence[str]) -> List[Tuple[str, int]]:
    specs: List[Tuple[str, int]] = []
    for asset in price_assets:
        specs.append((asset, _feature_index(feature_columns, asset)))
        specs.append((f"{asset}_volume", _feature_index(feature_columns, f"{asset}_volume")))
    return specs


def render_model_sanity(
    adapter: ModelAdapter,
    batch: StandardBatch,
    output_dir: Path,
    generation_length: int,
    seed: int,
    price_assets: Sequence[str] = DEFAULT_PRICE_ASSETS,
    num_simulations: int = NUM_SIMULATIONS,
) -> List[Path]:
    if batch.valid_windows is None or batch.valid_windows.shape[0] == 0:
        raise ValueError("Sanity visualization requires at least one validation window.")

    dl_set = load_dl_set(resolve_dl_set_path())
    feature_columns = dl_set["feature_columns"]
    stats = channel_norm_stats(dl_set)

    ground_truth = batch.valid_windows[0].float()
    if stats is not None:
        ground_truth = denormalize_channels(ground_truth.unsqueeze(0), *stats).squeeze(0)

    generated = adapter.generate(
        num_samples=num_simulations,
        generation_length=generation_length,
        seed=seed,
    )
    simulations = generated.data.float()
    if stats is not None:
        simulations = denormalize_channels(simulations, *stats)

    saved: List[Path] = []
    for label, channel_idx in channel_specs(price_assets, feature_columns):
        out_path = output_dir / f"{label}.png"
        _plot_channels(ground_truth, simulations, channel_idx, f"{label}: ground truth vs simulations", out_path)
        saved.append(out_path)
    return saved
