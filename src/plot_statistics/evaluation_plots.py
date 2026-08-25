"""Clean evaluation plotter — per-asset, QQ, relational, and downstream plots
for the StonkBench paper figures.

Generates:
  1. Per-asset: standalone QQ plots per channel (collected from evaluation)
  2. Relational: metric scatter plots, window size effects, DL vs stat comparison
  3. Downstream: hedging error distributions, portfolio optimization, P&L curves
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DEFAULT_DPI = 200
ASSET_PLOT_MAX = 25  # max assets per grid page


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_evaluation_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load complete_evaluation.json or scan for metrics.json files."""
    summary = results_dir / "complete_evaluation.json"
    if summary.exists():
        with open(summary) as f:
            return json.load(f)

    data: Dict[str, Dict[str, Any]] = {}
    for metrics_file in sorted(results_dir.glob("**/metrics.json")):
        with open(metrics_file) as f:
            data[metrics_file.parent.name] = json.load(f)
    return data


def _safe_get(d: Dict, key: str, default: float = np.nan) -> float:
    v = d.get(key, default)
    if isinstance(v, dict):
        v = v.get("diff", v.get("mean", default))
    return float(v) if v is not None else default


# Evaluator blocks in the on-disk metrics schema that hold scalar metric keys.
_METRIC_BLOCKS = ("FidelityEvaluator", "DiversityEvaluator", "StylizedFactsEvaluator")


def _nested_metric(d: Dict, key: str, default: float = np.nan) -> float:
    """Read a metric key from its nested evaluator block (real on-disk schema).

    The unified evaluator writes metrics under blocks (e.g. ``mdd`` lives in
    ``FidelityEvaluator``, ``icd_euclidean`` in ``DiversityEvaluator``). A
    top-level lookup alone returns NaN and silently produces empty plots.
    """
    for block in _METRIC_BLOCKS:
        v = d.get(block, {}).get(key)
        if v is not None:
            if isinstance(v, dict):
                v = v.get("diff", v.get("mean", v.get("real", default)))
            return float(v) if v is not None and np.isfinite(float(v)) else default
    return _safe_get(d, key, default)


# ---------------------------------------------------------------------------
# Plot 1: Per-Asset QQ + Distribution
# ---------------------------------------------------------------------------

def plot_per_asset(results: Dict[str, Dict], results_dir: Path, output_dir: Path) -> None:
    """Collect per-asset QQ plots written during evaluation into the plot tree.

    ``VisualAssessmentEvaluator`` saves one standalone ``qq_<channel>.png`` per
    asset under ``<results>/seq_<L>/<model>/per_asset/``. Copy them into
    ``plots/per_asset/<model>_seq<L>/`` so the paper figure folder is
    self-contained.
    """
    results_dir = Path(results_dir)
    copied = 0
    for key, data in results.items():
        if not isinstance(data, dict):
            continue
        model = data.get("model_name") or key.rsplit("_seq", 1)[0]
        seq = data.get("evaluated_at_length") or data.get("sequence_length")
        if seq is None:
            continue
        src = results_dir / f"seq_{seq}" / str(model) / "per_asset"
        if not src.exists():
            continue
        dst = output_dir / f"{model}_seq{seq}"
        dst.mkdir(parents=True, exist_ok=True)
        for png in sorted(src.glob("*.png")):
            shutil.copy2(png, dst / png.name)
            copied += 1
    if copied:
        print(f"  ✓ Per-asset QQ plots ({copied} files)")
    else:
        print("  [WARN] No per-asset QQ plots found; re-run eval with fixed VisualAssessmentEvaluator")

# ---------------------------------------------------------------------------
# Plot 2: Metric vs Metric Scatter (Fidelity vs Diversity, etc.)
# ---------------------------------------------------------------------------

def plot_metric_scatter(
    results: Dict[str, Dict],
    output_path: Path,
    x_metric: str = "mdd",
    y_metric: str = "icd_euclidean",
    x_label: str = "Marginal Distribution Distance",
    y_label: str = "ICD Euclidean",
) -> None:
    """Scatter plot of two metrics across models, colored by model class."""
    models = []
    x_vals, y_vals = [], []
    classes = {"deep_learning": [], "statistical": []}

    for model_name, data in results.items():
        if isinstance(data, dict) and model_name != "Real Data":
            x = _nested_metric(data, x_metric)
            y = _nested_metric(data, y_metric)
            if np.isfinite(x) and np.isfinite(y):
                x_vals.append(x)
                y_vals.append(y)
                models.append(model_name)
                mtype = data.get("model_type", "deep_learning")
                classes.setdefault(mtype, []).append(len(models) - 1)

    if not x_vals:
        print(f"[WARN] No valid data for {x_metric} vs {y_metric}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=DEFAULT_DPI)
    colors = {"deep_learning": "#1f77b4", "statistical": "#ff7f0e"}
    markers = {"deep_learning": "o", "statistical": "s"}

    for mtype, idxs in classes.items():
        xc = [x_vals[i] for i in idxs]
        yc = [y_vals[i] for i in idxs]
        ax.scatter(xc, yc, c=colors.get(mtype, "gray"), marker=markers.get(mtype, "o"),
                   label=mtype.replace("_", " ").title(), s=80, alpha=0.8, edgecolors="k")

    for i, name in enumerate(models):
        ax.annotate(name, (x_vals[i], y_vals[i]), fontsize=6, alpha=0.7,
                     textcoords="offset points", xytext=(3, 3))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs {x_label}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Window Size Effect (metric vs sequence length per model)
# ---------------------------------------------------------------------------

def plot_window_size_effect(
    results: Dict[str, Dict],
    output_dir: Path,
    metrics: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """For each metric, line plot showing effect of window size per model."""
    if metrics is None:
        metrics = [
            ("mdd", "Marginal Distribution Distance"),
            ("icd_euclidean", "ICD Euclidean"),
            ("autocorr_returns", "Autocorrelation Distance (diff)"),
            ("cmd", "Correlation Matrix Distance"),
        ]

    # Parse results into {model: {seq_len: {metric: val}}}
    model_seqs: Dict[str, Dict[int, Dict[str, float]]] = {}
    for key, data in results.items():
        if not isinstance(data, dict) or "sequence_length" not in data:
            continue
        model = data.get("model_name", key.rsplit("_seq", 1)[0])
        seq = int(data["sequence_length"])
        model_seqs.setdefault(model, {})[seq] = {
            m: _nested_metric(data, m) for m, _ in metrics
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric_key, metric_label in metrics:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=DEFAULT_DPI)
        for model, seq_dict in model_seqs.items():
            seqs = sorted(seq_dict.keys())
            vals = [seq_dict[s].get(metric_key, np.nan) for s in seqs]
            ax.plot(seqs, vals, marker="o", label=model, linewidth=2, markersize=6)

        ax.set_xlabel("Sequence Length (window size)")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Effect of Window Size on {metric_label}")
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"window_size_{metric_key}.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Plot 4: DL vs Statistical Radar / Grouped Bar
# ---------------------------------------------------------------------------

def plot_dl_vs_statistical(
    results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Grouped bar chart comparing DL vs statistical model average metrics."""
    metrics_config = [
        ("mdd", "MDD"),
        ("sdd", "SDD"),
        ("sd", "SD"),
        ("kd", "KD"),
        ("icd_euclidean", "ICD-Euc"),
        ("autocorr_returns", "ACD"),
        ("cmd", "CMD"),
        ("dcor_diff", "dCorDiff"),
    ]

    dl_vals: Dict[str, List[float]] = {}
    stat_vals: Dict[str, List[float]] = {}

    for model_name, data in results.items():
        if not isinstance(data, dict) or "model_type" not in data:
            continue
        mtype = data["model_type"]
        for mkey, _ in metrics_config:
            val = _nested_metric(data, mkey)
            if np.isfinite(val):
                if mtype == "deep_learning":
                    dl_vals.setdefault(mkey, []).append(val)
                else:
                    stat_vals.setdefault(mkey, []).append(val)

    if not dl_vals and not stat_vals:
        return

    fig, ax = plt.subplots(figsize=(12, 5), dpi=DEFAULT_DPI)
    n_metrics = len(metrics_config)
    x = np.arange(n_metrics)
    width = 0.35

    dl_means = [np.mean(dl_vals.get(m, [np.nan])) for m, _ in metrics_config]
    dl_stds = [np.std(dl_vals.get(m, [np.nan])) for m, _ in metrics_config]
    stat_means = [np.mean(stat_vals.get(m, [np.nan])) for m, _ in metrics_config]
    stat_stds = [np.std(stat_vals.get(m, [np.nan])) for m, _ in metrics_config]

    labels = [l for _, l in metrics_config]
    ax.bar(x - width/2, dl_means, width, yerr=dl_stds, label="Deep Learning",
           color="#1f77b4", capsize=3)
    ax.bar(x + width/2, stat_means, width, yerr=stat_stds, label="Statistical",
           color="#ff7f0e", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Metric Value (lower = better)")
    ax.set_title("DL vs Statistical Model Performance")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 5: Downstream — Hedging Error Distribution
# ---------------------------------------------------------------------------

def plot_hedging_error_distribution(
    results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Plot replication error summary across models (augmented testing)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, data in results.items():
        if not isinstance(data, dict):
            continue
        util = data.get("utility", {})
        aug = util.get("summary", {}).get("augmented_testing", {})
        if not aug:
            continue

        hedgers = list(aug.keys())
        real_means = [aug[h].get("real_train", {}).get("mean", 0) for h in hedgers]
        mixed_means = [aug[h].get("mixed_train", {}).get("mean", 0) for h in hedgers]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=DEFAULT_DPI)
        x = np.arange(len(hedgers))
        width = 0.35
        ax.bar(x - width/2, real_means, width, label="Real-only trained", color="#1f77b4")
        ax.bar(x + width/2, mixed_means, width, label="Mixed (Real+Synth) trained", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(hedgers, rotation=30, ha="right")
        ax.set_ylabel("Mean Replication Error")
        ax.set_title(f"{model_name}: Augmented Testing — Replication Error")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_dir / f"hedging_{model_name}.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Plot 6: Downstream — Portfolio Efficient Frontier
# ---------------------------------------------------------------------------

def plot_portfolio_comparison(
    results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Compare portfolio metrics across strategies and models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect Sharpe ratios per model per strategy
    all_sharpes: Dict[str, Dict[str, float]] = {}
    for model_name, data in results.items():
        if not isinstance(data, dict):
            continue
        port = data.get("portfolio", {})
        for n_assets_key, strategies in port.items():
            if not isinstance(strategies, dict):
                continue
            for strat_name, strat_data in strategies.items():
                if isinstance(strat_data, dict):
                    sr = strat_data.get("sharpe_ratio", np.nan)
                    all_sharpes.setdefault(strat_name, {})[model_name] = sr

    if not all_sharpes:
        return

    strategies = sorted(all_sharpes.keys())
    models = sorted(set().union(*(all_sharpes[s].keys() for s in strategies)))

    fig, ax = plt.subplots(figsize=(max(10, len(strategies) * 1.5), 6), dpi=DEFAULT_DPI)
    x = np.arange(len(models))
    width = 0.8 / len(strategies)

    for i, strat in enumerate(strategies):
        vals = [all_sharpes[strat].get(m, np.nan) for m in models]
        ax.bar(x + i * width, vals, width, label=strat)

    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Portfolio Optimization: Out-of-Sample Sharpe Ratio by Strategy")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "portfolio_sharpe_comparison.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 7: Downstream — P&L Curve Comparison
# ---------------------------------------------------------------------------

def plot_pnl_comparison(
    results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Compare P&L metrics across models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pnl_metrics = ["sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
                   "omega_ratio", "annualized_return", "annualized_volatility"]

    for metric_name in pnl_metrics:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=DEFAULT_DPI)
        models_list = []
        real_vals = []
        synth_vals = []

        for model_name, data in results.items():
            if not isinstance(data, dict):
                continue
            pnl = data.get("pnl", {})
            real_m = pnl.get("real", {}).get(metric_name, np.nan)
            synth_m = pnl.get("synthetic", {}).get(metric_name, np.nan)
            if np.isfinite(real_m) or np.isfinite(synth_m):
                models_list.append(model_name)
                real_vals.append(real_m)
                synth_vals.append(synth_m)

        if not models_list:
            continue

        x = np.arange(len(models_list))
        width = 0.35
        ax.bar(x - width/2, real_vals, width, label="Real Data", color="#1f77b4")
        ax.bar(x + width/2, synth_vals, width, label="Synthetic Data", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(models_list, rotation=30, ha="right")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"P&L: {metric_name.replace('_', ' ').title()} — Real vs Synthetic")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_dir / f"pnl_{metric_name}.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_plots(
    results_dir: Path,
    output_dir: Path,
    skip_heatmaps: bool = True,
) -> None:
    """Generate all paper-quality evaluation plots.

    Args:
        results_dir: Path containing complete_evaluation.json or metrics.json files.
        output_dir: Where to save generated plots.
        skip_heatmaps: Ignored (heatmaps already removed).
    """
    results = load_evaluation_results(results_dir)
    if not results:
        print(f"[ERROR] No evaluation results found in {results_dir}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    print(f"Generating plots from {len(results)} evaluation entries...")

    # 1. Per-asset QQ + distribution plots (collected from evaluation output)
    plot_per_asset(results, results_dir, output_dir / "per_asset")

    # 2. Metric scatter plots
    scatter_dir = output_dir / "scatter"
    plot_metric_scatter(results, scatter_dir / "fidelity_vs_diversity_mdd_icd.png",
                        "mdd", "icd_euclidean", "MDD", "ICD Euclidean")
    plot_metric_scatter(results, scatter_dir / "fidelity_vs_diversity_sdd_icd.png",
                        "sdd", "icd_euclidean", "SDD", "ICD Euclidean")
    plot_metric_scatter(results, scatter_dir / "cross_channel_cmd_vs_dcor.png",
                        "cmd", "dcor_diff", "CMD", "dCorDiff")
    print("  ✓ Metric scatter plots")

    # 3. Window size effects
    plot_window_size_effect(results, output_dir / "window_size")
    print("  ✓ Window size effect plots")

    # 4. DL vs Statistical comparison
    plot_dl_vs_statistical(results, output_dir / "comparison" / "dl_vs_statistical.png")
    print("  ✓ DL vs Statistical comparison")

    # 5. Hedging error distributions
    plot_hedging_error_distribution(results, output_dir / "downstream" / "hedging")
    print("  ✓ Hedging error plots")

    # 6. Portfolio comparison
    plot_portfolio_comparison(results, output_dir / "downstream" / "portfolio")
    print("  ✓ Portfolio optimization plots")

    # 7. P&L comparison
    plot_pnl_comparison(results, output_dir / "downstream" / "pnl")
    print("  ✓ P&L comparison plots")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation_plots")
    args = parser.parse_args()
    generate_all_plots(Path(args.results_dir), Path(args.output_dir))
