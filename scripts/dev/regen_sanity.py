"""Regenerate per-channel sanity plots for the kept run in outputs/.

For each of 14 final models this writes:
  outputs/sanity/<run>/<model>/<ticker>/overlay.png
  outputs/sanity/<run>/<model>/<ticker>/hist.png
  outputs/sanity/<run>/<model>/per_channel_summary.csv
  outputs/sanity/<run>/<model>/per_channel_summary.json
  outputs/sanity/<run>/<model>/manifest.json

We read the saved `.pt` artifact directly (we don't re-fit / re-generate) so
the plots bound the actual data on disk. Both simulated and ground-truth
data are denormalized using `dl_set` channel stats so the visualizations
are in raw log-return scale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path("/home/phamnhut/StonkBench")
DEFAULT_RUN_ID = "baseline_2026-07-21_vm"

# Canonical 14 final models (post 2026-07-23 cleanup + 2026-07-27 timegrad
# re-added).  timevae / sig_wgan / timegan are decommissioned and live
# under outputs_legacy2/ — kept out of the shipped peer bundle.
KEEP_DL = [
    "quantgan",
    "vrnn",
    "pcf_gan",
    "kalman_vae",
    "unconditional_tsdiffusion",
    "conditional_tsdiffusion",
    "cond_sig_wgan",
    "timegrad",
]
KEEP_STAT = [
    "gbm_adapter",
    "block_bootstrap",
    "ou_process",
    "merton_jump_diffusion",
    "de_jump_diffusion",
    "garch11",
]
ALL_14 = KEEP_DL + KEEP_STAT


def _load_artifact(model: str, results_dir: Path) -> tuple[torch.Tensor, dict]:
    """Return (data_tensor, metadata) for the canonical artifact path."""
    path = results_dir / model / "artifacts" / f"{model}_seq252.pt"
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "data" in obj and "metadata" in obj:
        return obj["data"].float(), obj["metadata"]
    # Legacy / non-payload-bearing artifact.
    if isinstance(obj, torch.Tensor):
        return obj.float(), {}
    raise ValueError(f"Unexpected artifact format at {path}: {type(obj)}")


def _load_ground_truth_window(results_dir: Path, model: str, seq_len: int = 252) -> torch.Tensor:
    """Load or derive ground truth for a model at given seq_len.

    Priority:
    1. Per-model GT file in the run directory (new multi-length format).
    2. Legacy single GT file.
    3. Build fresh GT from stats_set test_series using sliding_window_2d.
    """
    gt_path = results_dir / "ground_truth" / f"{model}_gt_seq{seq_len}.pt"
    if gt_path.is_file():
        obj = torch.load(gt_path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict) and "data" in obj:
            return obj["data"].float()
        if isinstance(obj, torch.Tensor):
            return obj.float()
    # Fallback: try legacy GT path
    legacy = ROOT / "outputs" / "data" / "ground_truth.pt"
    if legacy.is_file():
        obj = torch.load(legacy, map_location="cpu", weights_only=True)
        d = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
        return d[0].float()
    # Last resort: build GT from test series
    from src.utils.preprocessed_data_utils import load_stats_set, resolve_stats_set_path, sliding_window_2d
    stats = load_stats_set(resolve_stats_set_path())
    test_series = stats["test_series"].float()  # (T, C) raw log returns
    windows = sliding_window_2d(test_series, seq_len, stride=1)
    if windows.shape[0] == 0:
        raise FileNotFoundError(f"Cannot build GT: test_series too short for seq_len={seq_len}")
    return windows[0]  # (seq_len, C) first window


def _channel_metrics(
    gt_channel: np.ndarray, sim_channels: np.ndarray
) -> dict[str, float]:
    from scipy import stats as sp_stats

    gt_flat = gt_channel.reshape(-1)
    sim_flat = sim_channels.reshape(-1)
    try:
        ks_stat, ks_pval = sp_stats.ks_2samp(gt_flat, sim_flat)
    except Exception:
        ks_stat, ks_pval = float("nan"), 0.0
    if not np.isfinite(ks_stat):
        ks_stat, ks_pval = 0.0, 0.0
    return {
        "gt_mean": float(gt_flat.mean()),
        "gt_std": float(gt_flat.std() + 1e-12),
        "gt_skew": float(sp_stats.skew(gt_flat)) if gt_flat.size > 2 else 0.0,
        "gt_kurt": float(sp_stats.kurtosis(gt_flat)) if gt_flat.size > 3 else 0.0,
        "sim_mean": float(sim_flat.mean()),
        "sim_std": float(sim_flat.std() + 1e-12),
        "sim_skew": float(sp_stats.skew(sim_flat)) if sim_flat.size > 2 else 0.0,
        "sim_kurt": float(sp_stats.kurtosis(sim_flat)) if sim_flat.size > 3 else 0.0,
        "mean_diff": float(sim_flat.mean() - gt_flat.mean()),
        "std_ratio": float(sim_flat.std() / (gt_flat.std() + 1e-12)),
        "skew_diff": float(sp_stats.skew(sim_flat) - sp_stats.skew(gt_flat))
        if sim_flat.size > 2 and gt_flat.size > 2
        else 0.0,
        "kurt_diff": float(sp_stats.kurtosis(sim_flat) - sp_stats.kurtosis(gt_flat))
        if sim_flat.size > 3 and gt_flat.size > 3
        else 0.0,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pval),
    }


def _plot_overlay(
    gt_window: torch.Tensor,
    simulations: torch.Tensor,
    channel_idx: int,
    channel_name: str,
    output_path: Path,
) -> None:
    gt = gt_window[:, channel_idx].detach().cpu().numpy()
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
    gt_window: torch.Tensor,
    simulations: torch.Tensor,
    channel_idx: int,
    channel_name: str,
    output_path: Path,
) -> None:
    gt = gt_window[:, channel_idx].detach().cpu().numpy().reshape(-1)
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


def regen_for_model(
    model: str,
    feature_columns: list[str],
    gt_window: torch.Tensor,
    skip_existing: bool,
    results_dir: Path,
    sanity_dir: Path,
) -> dict:
    """Regen per-ticker plots + summary CSV/JSON for one model.

    Returns a dict of {channel -> [overlay_path, hist_path]}.
    """
    sims, meta = _load_artifact(model, results_dir)
    if sims.ndim != 3:
        raise ValueError(f"Expected 3D artifact for {model}; got {sims.shape}")
    if sims.shape[-1] != len(feature_columns):
        # Truncate or pick min to be safe.
        n = min(sims.shape[-1], len(feature_columns))
        sims = sims[..., :n]
        feature_columns = feature_columns[:n]

    out_dir = sanity_dir / model
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    rows: list[dict] = []

    for ch_idx, ch_name in enumerate(feature_columns):
        overlay_path = out_dir / ch_name / "overlay.png"
        hist_path = out_dir / ch_name / "hist.png"
        summary_present = (out_dir / "per_channel_summary.csv").exists()
        if (
            skip_existing
            and overlay_path.exists()
            and hist_path.exists()
            and summary_present
        ):
            # Treat the entire model directory as already done.
            saved.extend([overlay_path, hist_path])
            continue
        _plot_overlay(gt_window, sims, ch_idx, ch_name, overlay_path)
        _plot_histogram(gt_window, sims, ch_idx, ch_name, hist_path)
        saved.extend([overlay_path, hist_path])
        rows.append(
            {"channel": ch_name, "channel_idx": ch_idx, **_channel_metrics(
                gt_window[:, ch_idx].detach().cpu().numpy(),
                sims[:, :, ch_idx].detach().cpu().numpy(),
            )}
        )

    # Per-channel summary
    if rows:
        import csv

        csv_path = out_dir / "per_channel_summary.csv"
        json_path = out_dir / "per_channel_summary.json"
        fields = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        with json_path.open("w", encoding="utf-8") as f:
            import json as _json

            _json.dump(
                {"channel_count": len(rows), "channels": rows}, f, indent=2
            )
        saved.extend([csv_path, json_path])

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists() or not skip_existing:
        import json as _json

        manifest = {
            "model_name": model,
            "sequence_length": int(sims.shape[1]),
            "num_simulations": int(sims.shape[0]),
            "channel_count": len(feature_columns),
            "channels": feature_columns,
            "regenerated_by": "scripts/dev/regen_sanity.py",
            "artifact_path": str(
                results_dir / model / "artifacts" / f"{model}_seq252.pt"
            ),
            "ground_truth_path": "(per-model GT in run dir)",
        }
        manifest_path.write_text(_json.dumps(manifest, indent=2))
        saved.append(manifest_path)

    return {"model": model, "files": [str(p) for p in saved]}


def verify_only(results_dir: Path, sanity_dir: Path) -> int:
    """Print a status table; return exit code (0 if all OK)."""
    print("=== verify (no writes) ===")
    rc = 0
    print(f'  {"model":<28} {"artifact":<10} {"sanity":<10} {"pngs":>5}')
    print("  " + "-" * 56)
    for m in ALL_14:
        art_p = results_dir / m / "artifacts" / f"{m}_seq252.pt"
        san_d = sanity_dir / m
        art_ok = "MISS"
        if art_p.is_file():
            art_ok = "OK"
        san_ok = "OK" if san_d.is_dir() else "MISS"
        png_count = 0
        if san_d.is_dir():
            png_count = sum(1 for _ in san_d.rglob("*.png"))
        if art_ok != "OK" or san_ok != "OK" or png_count < 25:
            rc = 1
        print(f"  {m:<28} {art_ok:<10} {san_ok:<10} {png_count:>5}")
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="leave existing PNGs and CSVs alone; only fill missing pieces",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="just print status table without writing anything",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="subset of canonical 13 to regen (default: all available)",
    )
    ap.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=f"run directory under outputs/results/ (default: {DEFAULT_RUN_ID})",
    )
    args = ap.parse_args()
    run_id = args.run_id

    if args.verify_only:
        results_dir_v = ROOT / "outputs" / "results" / run_id
        sanity_dir_v = ROOT / "outputs" / "sanity" / run_id
        raise SystemExit(verify_only(results_dir_v, sanity_dir_v))

    # Channel names (asset columns) come from the preprocessed dl_set.
    sys.path.insert(0, str(ROOT))
    from src.utils.preprocessed_data_utils import load_dl_set, resolve_dl_set_path

    dl_set = load_dl_set(resolve_dl_set_path())
    feature_columns = list(dl_set["feature_columns"])

    results_dir = ROOT / "outputs" / "results" / run_id
    sanity_dir = ROOT / "outputs" / "sanity" / run_id

    targets = args.models if args.models else ALL_14
    summary: list[dict] = []
    for model in targets:
        art_p = results_dir / model / "artifacts" / f"{model}_seq252.pt"
        if not art_p.is_file():
            print(f"  SKIP  {model}: artifact missing at {art_p}")
            continue
        try:
            gt_window = _load_ground_truth_window(results_dir, model, 252)
            result = regen_for_model(model, feature_columns, gt_window, args.skip_existing, results_dir, sanity_dir)
            print(
                f"  WROTE {model:<28} -> {len(result['files'])} files"
            )
            summary.append(result)
        except Exception as exc:
            print(f"  ERR   {model}: {exc}")
    print()
    print(f"=== regenerated {len(summary)} of {len(targets)} requested models ===")
    raise SystemExit(0 if summary else 1)


if __name__ == "__main__":
    main()
