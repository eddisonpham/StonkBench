"""Mini-experiment logger — logs every DL model's training + fidelity outputs.

Design goals (user request): log ALL outputs simultaneously into one run dir,
and make them monitorable live while jobs run.

Layout of a run directory ``<out_root>/<run_id>/``::

    events/<model>.jsonl          live per-epoch train/val losses (tailable)
    fit_summary.jsonl             one line per model after fit()
    fidelity.jsonl                one line per (model, seq_len) fidelity metrics
    curves/<model>.csv            epoch,train_loss,val_loss (post-fit, tidy)
    plots/<model>_curves.png      loss curves
    plots/<model>_seq<L>.png      fidelity bars (std ratio/kurt) + sample paths
    plots/all_curves.png          grid of every model's live curves
    report.md                     aggregated verdict table

Everything here is append-only and idempotent-ish: refresh_report() can be
re-run any time (e.g. mid-run) and rebuilds report.md + all_curves.png.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

# Non-interactive backend MUST be selected before pyplot import.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = [
    "MiniLogger",
    "compute_fidelity",
    "load_jsonl",
    "refresh_report",
]


# --------------------------------------------------------------------------
# low-level JSONL helpers
# --------------------------------------------------------------------------
def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.time(), **payload}
    # O_APPEND single small write — safe for concurrent writers.
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn last line mid-write
    return rows


def _finite(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
# fidelity metrics (raw space: gen (N,L,C), gt (M,L,C))
# --------------------------------------------------------------------------
def compute_fidelity(gen: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    """Distribution-fidelity metrics of generated windows vs ground truth.

    Returns scalars (logged + tabulated) plus per-channel arrays (plotted).
    """
    gen = np.asarray(gen, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    n_gen, n_gt = gen.shape[0], gt.shape[0]
    flat_gen = gen.reshape(-1, gen.shape[-1])
    flat_gt = gt.reshape(-1, gt.shape[-1])

    nonfinite = float(1.0 - np.isfinite(gen).mean())
    gen_f = _finite(flat_gen)
    gt_f = _finite(flat_gt)

    std_g = gen_f.std(axis=0)
    std_t = gt_f.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        std_ratio = np.where(std_t > 1e-12, std_g / std_t, np.nan)
    std_ratio_f = _finite(std_ratio)

    mean_abs_diff = float(np.abs(gen_f.mean(axis=0) - gt_f.mean(axis=0)).mean())

    def _kurt(x: np.ndarray) -> np.ndarray:
        m = x.mean(axis=0)
        c = x - m
        s2 = (c ** 2).mean(axis=0)
        s4 = (c ** 4).mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return _finite(np.where(s2 > 1e-18, s4 / (s2 ** 2) - 3.0, 0.0))

    kurt_g, kurt_t = _kurt(gen_f), _kurt(gt_f)

    def _lag1(x: np.ndarray) -> np.ndarray:
        a, b = x[:-1], x[1:]
        a = a - a.mean(axis=0)
        b = b - b.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.sqrt((a ** 2).mean(axis=0) * (b ** 2).mean(axis=0))
            return _finite(np.where(denom > 1e-18, (a * b).mean(axis=0) / denom, 0.0))

    lag1_g, lag1_t = _lag1(gen_f), _lag1(gt_f)

    def _xcorr(x: np.ndarray) -> np.ndarray:
        xc = np.corrcoef(x, rowvar=False)
        return _finite(np.atleast_2d(xc))

    xc_g, xc_t = _xcorr(gen_f), _xcorr(gt_f)
    corr_rel_frob = float(
        np.linalg.norm(xc_g - xc_t) / max(np.linalg.norm(xc_t), 1e-12)
    )

    # Per-channel two-sample KS (marginal distribution shape).
    try:
        from scipy.stats import ks_2samp

        ks = [
            float(ks_2samp(gen_f[:, c], gt_f[:, c]).statistic)
            for c in range(gen_f.shape[1])
        ]
        ks_mean = float(np.mean(ks))
    except Exception:  # noqa: BLE001 — scipy optional for this diagnostic
        ks, ks_mean = [], float("nan")

    healthy = np.isfinite(std_ratio) & (std_ratio >= 0.8) & (std_ratio <= 1.25)

    return {
        "n_gen": int(n_gen),
        "n_gt": int(n_gt),
        "nonfinite_frac": nonfinite,
        "std_ratio_mean": float(np.nanmean(std_ratio_f)),
        "std_ratio_min": float(np.nanmin(std_ratio_f)),
        "std_ratio_max": float(np.nanmax(std_ratio_f)),
        "std_ratio_healthy_frac": float(healthy.mean()),
        "mean_abs_diff": mean_abs_diff,
        "kurt_abs_diff": float(np.abs(kurt_g - kurt_t).mean()),
        "gt_kurt_mean": float(kurt_t.mean()),
        "gen_kurt_mean": float(kurt_g.mean()),
        "lag1_abs_diff": float(np.abs(lag1_g - lag1_t).mean()),
        "corr_rel_frob": corr_rel_frob,
        "ks_mean": ks_mean,
        "std_ratio_per_channel": [float(x) for x in std_ratio_f],
        "kurt_gen_per_channel": [float(x) for x in kurt_g],
        "kurt_gt_per_channel": [float(x) for x in kurt_t],
    }


# --------------------------------------------------------------------------
# per-run logger
# --------------------------------------------------------------------------
class MiniLogger:
    """One instance per (run_dir, model). All paths live under the run dir."""

    def __init__(self, run_dir: Path, model: str):
        self.run_dir = Path(run_dir)
        self.model = model
        (self.run_dir / "events").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "curves").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)

    # -- events ------------------------------------------------------------
    def event(self, kind: str, **fields: Any) -> None:
        _append_jsonl(
            self.run_dir / "events" / f"{self.model}.jsonl",
            {"event": kind, "model": self.model, **fields},
        )

    # -- fit ---------------------------------------------------------------
    def log_fit(self, fit_info: Dict[str, Any], elapsed_sec: float) -> None:
        summary = {
            k: v
            for k, v in fit_info.items()
            if not isinstance(v, (list, dict))
        }
        summary["wall_clock_seconds"] = float(elapsed_sec)
        _append_jsonl(self.run_dir / "fit_summary.jsonl", {"model": self.model, **summary})
        self.event("fit_complete", **summary)

        # tidy per-epoch CSV (epoch- or step-indexed)
        train_hist = [float(x) for x in fit_info.get("train_loss_history", []) or []]
        val_hist = [float(x) for x in fit_info.get("val_loss_history", []) or []]
        if train_hist or val_hist:
            csv_path = self.run_dir / "curves" / f"{self.model}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["step", "train_loss", "val_loss"])
                n = max(len(train_hist), len(val_hist))
                for i in range(n):
                    writer.writerow(
                        [
                            i + 1,
                            train_hist[i] if i < len(train_hist) else "",
                            val_hist[i] if i < len(val_hist) else "",
                        ]
                    )

    # -- step curves (cond_sig_wgan) ---------------------------------------
    def log_step_losses(self, step_losses: Dict[str, List[float]]) -> None:
        if not step_losses:
            return
        keys = sorted(step_losses.keys())
        csv_path = self.run_dir / "curves" / f"{self.model}_steps.csv"
        n = max(len(step_losses[k]) for k in keys)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step"] + keys)
            for i in range(n):
                writer.writerow(
                    [i + 1] + [step_losses[k][i] if i < len(step_losses[k]) else "" for k in keys]
                )
        self.event("step_losses", n_steps=n, keys=keys)

    # -- fidelity ----------------------------------------------------------
    def log_fidelity(self, seq_len: int, metrics: Dict[str, Any]) -> None:
        flat = {k: v for k, v in metrics.items() if not isinstance(v, list)}
        _append_jsonl(
            self.run_dir / "fidelity.jsonl",
            {"model": self.model, "seq_len": int(seq_len), **flat},
        )
        self.event("fidelity", seq_len=int(seq_len), **flat)

    # -- plots -------------------------------------------------------------
    def plot_curves(self, fit_info: Dict[str, Any]) -> Optional[Path]:
        train_hist = [float(x) for x in fit_info.get("train_loss_history", []) or []]
        val_hist = [float(x) for x in fit_info.get("val_loss_history", []) or []]
        step_csv = self.run_dir / "curves" / f"{self.model}_steps.csv"
        if not (train_hist or val_hist) and not step_csv.exists():
            return None
        fig, ax = plt.subplots(figsize=(7, 4))
        if train_hist:
            ax.plot(range(1, len(train_hist) + 1), train_hist, label="train", lw=1.2)
        if val_hist:
            ax.plot(range(1, len(val_hist) + 1), val_hist, label="val", lw=1.2)
        if not train_hist and not val_hist and step_csv.exists():
            with open(step_csv, "r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for key in ("loss", "sig_w1", "total_norm"):
                if key in (rows[0] if rows else {}):
                    ys = [float(r[key]) for r in rows if r.get(key)]
                    ax.plot(range(1, len(ys) + 1), ys, label=key, lw=0.8)
        ax.set_xlabel("epoch / step")
        ax.set_ylabel("loss")
        ax.set_title(f"{self.model}: training curves")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        out = self.run_dir / "plots" / f"{self.model}_curves.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return out

    def plot_fidelity(
        self,
        seq_len: int,
        metrics: Dict[str, Any],
        gen: np.ndarray,
        gt: np.ndarray,
        asset_names: Sequence[str],
        n_path_channels: int = 3,
    ) -> Optional[Path]:
        std_ratio = np.asarray(metrics["std_ratio_per_channel"], dtype=float)
        kurt_g = np.asarray(metrics["kurt_gen_per_channel"], dtype=float)
        kurt_t = np.asarray(metrics["kurt_gt_per_channel"], dtype=float)
        if std_ratio.size == 0:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))

        ax = axes[0][0]
        ax.bar(range(len(std_ratio)), std_ratio, color="steelblue")
        ax.axhspan(0.8, 1.25, color="green", alpha=0.12, label="healthy 0.8-1.25")
        ax.axhline(1.0, color="black", lw=0.8)
        ax.set_title(f"{self.model} seq{seq_len}: std ratio (gen / GT)")
        ax.set_xticks([])
        ax.legend(fontsize=7)

        ax = axes[0][1]
        ax.bar(range(len(kurt_t)), kurt_t, color="grey", alpha=0.6, label="GT")
        ax.bar(range(len(kurt_g)), kurt_g, color="indianred", alpha=0.7, label="gen")
        ax.set_title(f"excess kurtosis (mean |diff|={metrics['kurt_abs_diff']:.2f})")
        ax.set_xticks([])
        ax.legend(fontsize=7)

        # sample paths vs GT envelope for first channels
        n_show = min(n_path_channels, gen.shape[-1])
        for i in range(n_show):
            ax = axes[1][0] if i == 0 else axes[1][1]
            if i >= 2:
                break
            gt_t = gt[:, :, i]  # (M, L)
            gn_t = gen[:, :, i]
            L = gt_t.shape[1]
            gt_mean = gt_t.mean(axis=0)
            gt_lo, gt_hi = np.percentile(gt_t, [5, 95], axis=0)
            gn_mean = gn_t.mean(axis=0)
            gn_lo, gn_hi = np.percentile(gn_t, [5, 95], axis=0)
            x = np.arange(L)
            ax.fill_between(x, gt_lo, gt_hi, color="grey", alpha=0.35, label="GT 5-95%")
            ax.plot(x, gt_mean, color="black", lw=1.2, label="GT mean")
            ax.fill_between(x, gn_lo, gn_hi, color="tomato", alpha=0.3, label="gen 5-95%")
            ax.plot(x, gn_mean, color="tomato", lw=1.2, label="gen mean")
            name = asset_names[i] if i < len(asset_names) else f"ch{i}"
            ax.set_title(f"{name}: path spread")
            ax.set_xlabel("t")
            ax.legend(fontsize=7)
        if n_show == 1:
            axes[1][1].axis("off")

        fig.suptitle(
            f"{self.model} seq{seq_len} | std_ratio {metrics['std_ratio_mean']:.2f} "
            f"healthy {metrics['std_ratio_healthy_frac']*100:.0f}% | "
            f"KS {metrics['ks_mean']:.3f} | corr-diff {metrics['corr_rel_frob']:.3f} | "
            f"nonfinite {metrics['nonfinite_frac']:.3f}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = self.run_dir / "plots" / f"{self.model}_seq{seq_len}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return out


# --------------------------------------------------------------------------
# run-level aggregation (usable mid-run)
# --------------------------------------------------------------------------
def refresh_report(run_dir: Path) -> Path:
    """Rebuild report.md + plots/all_curves.png from whatever is on disk."""
    run_dir = Path(run_dir)
    # fair-train runs never instantiate MiniLogger, so plots/ may not exist yet.
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    # --- live curves grid from events/*.jsonl ---
    # Rows are grouped by their ``model`` field (one file may carry several
    # models, e.g. fair_cheap.jsonl; per-asset QuantGAN rows like
    # ``quantgan.ch3`` collapse to their base model and aggregate per epoch).
    events_dir = run_dir / "events"
    model_events: Dict[str, List[Dict[str, Any]]] = {}
    if events_dir.exists():
        for path in sorted(events_dir.glob("*.jsonl")):
            for row in load_jsonl(path):
                if row.get("event") != "epoch":
                    continue
                key = str(row.get("model", path.stem))
                key = re.sub(r"\.ch\d+$", "", key)
                model_events.setdefault(key, []).append(row)

    if model_events:
        n = len(model_events)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), squeeze=False)
        for idx, (model, rows) in enumerate(sorted(model_events.items())):
            ax = axes[idx // ncols][idx % ncols]
            rows = sorted(rows, key=lambda r: (r.get("epoch", 0), r.get("ts", 0)))
            # per-asset quantgan events: aggregate by epoch
            by_epoch: Dict[int, List[Dict[str, Any]]] = {}
            for r in rows:
                by_epoch.setdefault(int(r.get("epoch", 0)), []).append(r)
            epochs = sorted(by_epoch)
            tr, va = [], []
            for e in epochs:
                group = by_epoch[e]
                tvals = [g["train_loss"] for g in group if g.get("train_loss") is not None]
                vvals = [g["val_loss"] for g in group if g.get("val_loss") is not None]
                tr.append(float(np.mean(tvals)) if tvals else np.nan)
                va.append(float(np.mean(vvals)) if vvals else np.nan)
            ax.plot(epochs, tr, label="train", lw=1.1)
            if any(not math.isnan(v) for v in va):
                ax.plot(epochs, va, label="val", lw=1.1)
            ax.set_title(model, fontsize=9)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        for idx in range(len(model_events), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        fig.suptitle(f"{run_dir.name}: live training curves", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(run_dir / "plots" / "all_curves.png", dpi=110)
        plt.close(fig)

    # --- fidelity table ---
    fid_rows = load_jsonl(run_dir / "fidelity.jsonl")
    fit_rows = load_jsonl(run_dir / "fit_summary.jsonl")
    fit_by_model = {r.get("model"): r for r in fit_rows}

    lines: List[str] = []
    lines.append(f"# Mini DL experiment report — {run_dir.name}")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Fidelity vs ground truth (denormalized raw returns)")
    lines.append("")
    lines.append(
        "| model | seq | n_gen | n_gt | std_ratio [min..max] | healthy% | kurt diff "
        "| lag1 diff | corr_diff | KS | nonfinite |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(fid_rows, key=lambda r: (str(r.get("model")), int(r.get("seq_len", 0)))):
        lines.append(
            "| {model} | {seq} | {n_gen} | {n_gt} | {mean:.2f} [{lo:.2f}..{hi:.2f}] "
            "| {healthy:.0f}% | {kurt:.2f} | {lag:.3f} | {corr:.3f} | {ks:.3f} "
            "| {nf:.3f} |".format(
                model=r.get("model"),
                seq=r.get("seq_len"),
                n_gen=r.get("n_gen"),
                n_gt=r.get("n_gt"),
                mean=float(r.get("std_ratio_mean", float("nan"))),
                lo=float(r.get("std_ratio_min", float("nan"))),
                hi=float(r.get("std_ratio_max", float("nan"))),
                healthy=100.0 * float(r.get("std_ratio_healthy_frac", 0.0)),
                kurt=float(r.get("kurt_abs_diff", float("nan"))),
                lag=float(r.get("lag1_abs_diff", float("nan"))),
                corr=float(r.get("corr_rel_frob", float("nan"))),
                ks=float(r.get("ks_mean", float("nan"))),
                nf=float(r.get("nonfinite_frac", float("nan"))),
            )
        )
    lines.append("")
    lines.append("## Fit summaries")
    lines.append("")
    lines.append("| model | best_val | best_epoch | stopped_early | wall_s |")
    lines.append("|---|---|---|---|---|")
    for model, r in sorted(fit_by_model.items()):
        lines.append(
            "| {m} | {bv} | {be} | {se} | {w:.1f} |".format(
                m=model,
                bv=r.get("best_val_loss"),
                be=r.get("best_epoch"),
                se=r.get("stopped_early"),
                w=float(r.get("wall_clock_seconds", 0.0)),
            )
        )
    lines.append("")
    lines.append("Plots: `plots/<model>_curves.png`, `plots/<model>_seq<L>.png`, `plots/all_curves.png`")
    lines.append("")

    report = run_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
