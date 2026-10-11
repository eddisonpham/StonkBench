#!/usr/bin/env python
"""Score generated sequences for collapse / realism — NOT a downstream eval.

Scans ``<output_root>/results/<run_id>/<model>/artifacts/*.pt``, compares each
generated tensor against ground truth rebuilt from the dl_set (exactly what
the unified evaluator uses), and emits per-channel collapse flags:

  COLLAPSED channel : std_ratio < 0.1 or gen std < 1e-6 (variance starvation)
  NONFINITE         : any NaN/Inf in the generated tensor
  FLAT              : every value identical
  TAIL_WARN         : |excess kurtosis difference| > 10 (unrealistic tails)
  DISP_WARN         : mean std_ratio outside [0.8, 1.25]

Verdict per (model, seq): COLLAPSED (>=25% channels collapsed) /
WARN (any flag) / PASS.

Writes ``<report_dir>/sanity_outputs.md`` + ``sanity_outputs.json`` and
prints the markdown to stdout. Safe to run while other models are still
training (partial results) and re-run any time.

Usage:
    python scripts/mini_exp/score_outputs.py \
        --run_id fair_20261004 \
        --output_root /home/epham/StonkBench/outputs \
        --report_dir /home/epham/StonkBench/outputs/mini_exp/fair_20261004
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from exp_logger import compute_fidelity  # noqa: E402

from src.utils.env import get_output_root  # noqa: E402
from src.utils.preprocessed_data_utils import (  # noqa: E402
    channel_norm_stats,
    denormalize_channels,
    load_dl_set,
    resolve_dl_set_path,
    sliding_window_2d,
)

COLLAPSED_RATIO = 0.1          # std_ratio below this => collapsed channel
COLLAPSED_FRAC = 0.25          # >= this fraction of channels collapsed => COLLAPSED
FLAT_STD = 1e-6
KURT_WARN = 10.0
DISP_LO, DISP_HI = 0.8, 1.25


def load_artifact(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj.get("data", obj)
    arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected (N,L,C) array, got {arr.shape}")
    return arr


def collapse_flags(gen: np.ndarray, gt: np.ndarray, metrics: dict) -> dict:
    """Per-channel collapse/realism flags for one (model, seq) tensor."""
    n_chan = gen.shape[-1]
    flat_g = gen.reshape(-1, n_chan)
    gen_std = flat_g.std(axis=0)
    ratios = np.asarray(metrics["std_ratio_per_channel"], dtype=float)

    collapsed = (ratios < COLLAPSED_RATIO) | (gen_std < FLAT_STD)
    # flat = every value identical in that channel
    flat = gen_std < FLAT_STD
    nonfinite = float(1.0 - np.isfinite(gen).mean())
    kurt_diff = float(metrics["kurt_abs_diff"])
    mean_ratio = float(metrics["std_ratio_mean"])

    flags: list[str] = []
    n_collapsed = int(collapsed.sum())
    if n_collapsed:
        idx = [int(i) for i in np.where(collapsed)[0][:10]]
        flags.append(f"COLLAPSED_CHANNELS({n_collapsed}/{n_chan} e.g. ch{idx})")
    if nonfinite > 0:
        flags.append(f"NONFINITE({nonfinite:.4f})")
    if int(flat.sum()):
        flags.append(f"FLAT({int(flat.sum())})")
    if kurt_diff > KURT_WARN:
        flags.append(f"TAIL_WARN(kurt_diff={kurt_diff:.1f})")
    if not (DISP_LO <= mean_ratio <= DISP_HI):
        flags.append(f"DISP_WARN(std_ratio={mean_ratio:.2f})")

    if n_collapsed >= max(1, int(COLLAPSED_FRAC * n_chan)) or nonfinite > 0:
        verdict = "COLLAPSED"
    elif flags:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "flags": flags,
        "n_channels": n_chan,
        "n_collapsed": n_collapsed,
        "collapsed_idx": [int(i) for i in np.where(collapsed)[0]],
        "nonfinite_frac": nonfinite,
        "std_ratio_mean": mean_ratio,
        "std_ratio_min": float(np.nanmin(ratios)),
        "healthy_frac": float(metrics["std_ratio_healthy_frac"]),
        "kurt_abs_diff": kurt_diff,
        "lag1_abs_diff": float(metrics["lag1_abs_diff"]),
        "corr_rel_frob": float(metrics["corr_rel_frob"]),
        "ks_mean": float(metrics["ks_mean"]),
        "n_gen": int(gen.shape[0]),
        "n_gt": int(gt.shape[0]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_id", required=True)
    p.add_argument("--output_root", default=None,
                   help="default: $STONKBENCH_OUTPUT_ROOT or <repo>/outputs")
    p.add_argument("--report_dir", default=None,
                   help="default: /home/epham/StonkBench/outputs/mini_exp/<run_id>")
    p.add_argument("--seq_lengths", default="21,42,126,252")
    args = p.parse_args()

    output_root = Path(args.output_root) if args.output_root else get_output_root()
    results_dir = output_root / "results" / args.run_id
    report_dir = Path(args.report_dir) if args.report_dir else (
        get_output_root() / "mini_exp" / args.run_id
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    seq_lengths = [int(x) for x in args.seq_lengths.split(",") if x.strip()]

    if not results_dir.exists():
        print(f"ERROR: no results at {results_dir}", file=sys.stderr)
        return 2

    # Ground truth rebuilt exactly like the pipeline/evaluator: denormalized
    # test series -> sliding windows.
    dl_set = load_dl_set(resolve_dl_set_path())
    stats = channel_norm_stats(dl_set)
    if stats is None:
        print("ERROR: dl_set has no channel norm stats", file=sys.stderr)
        return 2
    mean, std = stats
    test_raw = denormalize_channels(dl_set["test_series"].float(), mean, std)
    gt_by_seq = {
        L: sliding_window_2d(test_raw, L, stride=1).numpy() for L in seq_lengths
    }

    rows = []
    fidelity_rows = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "ground_truth":
            continue
        art_dir = model_dir / "artifacts"
        if not art_dir.exists():
            continue
        for seq_len in seq_lengths:
            art = art_dir / f"{model_dir.name}_seq{seq_len}.pt"
            if not art.exists():
                continue
            gt = gt_by_seq[seq_len]
            if gt.shape[0] == 0:
                continue
            try:
                gen = load_artifact(art)
                meta = {}
                raw = torch.load(art, map_location="cpu", weights_only=False)
                if isinstance(raw, dict):
                    meta = raw.get("metadata", {}) or {}
                metrics = compute_fidelity(gen, gt)
                # full fidelity metrics -> report.md table via refresh_report
                fidelity_rows.append({
                    "model": model_dir.name, "seq_len": seq_len, **metrics,
                })
                row = {
                    "model": model_dir.name,
                    "seq_len": seq_len,
                    "best_epoch": meta.get("best_epoch"),
                    "best_val_loss": meta.get("best_val_loss"),
                    "stopped_early": meta.get("stopped_early"),
                    "wall_clock_seconds": meta.get("wall_clock_seconds"),
                    **collapse_flags(gen, gt, metrics),
                }
            except Exception as exc:  # noqa: BLE001 — report, don't crash
                row = {"model": model_dir.name, "seq_len": seq_len,
                       "verdict": "ERROR", "flags": [f"EXC({exc})"]}
            rows.append(row)
            print(
                f"{row['verdict']:9s} {row['model']:30s} seq{seq_len:<4d} "
                f"std_ratio={row.get('std_ratio_mean', float('nan')):.3f} "
                f"min={row.get('std_ratio_min', float('nan')):.3f} "
                f"healthy={100*row.get('healthy_frac', 0):3.0f}% "
                f"collapsed={row.get('n_collapsed', 0)}/{row.get('n_channels', 0)} "
                f"flags={','.join(row.get('flags', [])) or '-'}",
                flush=True,
            )

    (report_dir / "sanity_outputs.json").write_text(json.dumps(rows, indent=1))

    # fidelity.jsonl (same schema MiniLogger.log_fidelity writes) so
    # refresh_report's fidelity table is populated for pipeline runs too.
    if fidelity_rows:
        with (report_dir / "fidelity.jsonl").open("w", encoding="utf-8") as fh:
            for r in fidelity_rows:
                fh.write(json.dumps(r) + "\n")

    # fit_summary.jsonl from artifact metadata (best_val_loss/best_epoch/
    # stopped_early/wall_s are stored in every artifact's metadata).
    fit_by_model: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        model = r.get("model")
        if not model or model in fit_by_model:
            continue
        fit_by_model[model] = {
            "model": model,
            "best_val_loss": r.get("best_val_loss"),
            "best_epoch": r.get("best_epoch"),
            "stopped_early": r.get("stopped_early"),
            "wall_clock_seconds": r.get("wall_clock_seconds"),
        }
    if fit_by_model:
        with (report_dir / "fit_summary.jsonl").open("w", encoding="utf-8") as fh:
            for r in sorted(fit_by_model.values(), key=lambda r: str(r["model"])):
                fh.write(json.dumps(r) + "\n")

    # markdown report
    lines = [
        f"# Output sanity report — {args.run_id}",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  results: `{results_dir}`",
        "",
        "PASS = finite, no collapsed channels, dispersion in [0.8,1.25], "
        "kurtosis diff <= 10. WARN/COLLAPSED explains the flag list.",
        "",
        "| model | seq | verdict | std_ratio (min) | healthy% | collapsed | best_epoch | wall_s | flags |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {model} | {seq} | **{v}** | {sr:.2f} ({srmin:.2f}) | {h:.0f}% "
            "| {c}/{n} | {be} | {w} | {f} |".format(
                model=r["model"], seq=r["seq_len"], v=r["verdict"],
                sr=r.get("std_ratio_mean", float("nan")),
                srmin=r.get("std_ratio_min", float("nan")),
                h=100 * r.get("healthy_frac", 0),
                c=r.get("n_collapsed", "-"), n=r.get("n_channels", "-"),
                be=r.get("best_epoch", "-"),
                w=round(r["wall_clock_seconds"], 0) if r.get("wall_clock_seconds") else "-",
                f="; ".join(r.get("flags", [])) or "-",
            )
        )
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    lines += ["", "Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), ""]
    report = report_dir / "sanity_outputs.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {report}")
    print("\n".join(lines))
    # non-zero if anything collapsed so job logs show it plainly
    bad = sum(1 for r in rows if r["verdict"] in ("COLLAPSED", "ERROR"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
