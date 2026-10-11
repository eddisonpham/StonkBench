"""Aggregate the 3 replicate evaluation runs per ablation into mean ± std.

Reads one ``complete_evaluation.json`` per evaluated cell:

    <root>/<ablation>/evaluation_batch{1,2,3}/<model>_seq<L>/complete_evaluation.json

``scripts/mini_exp/eval_gen_batches.sh`` writes exactly that layout (one array
task per artifact, each into its own results directory — the evaluator
overwrites ``complete_evaluation.json`` on every run, so per-cell directories
are what makes the 432 tasks safe to run concurrently). Older trees that kept a
single batch-level summary are still accepted as a fallback.

Every numeric metric of every ``<model>_seq<L>`` cell is flattened so the
batch-to-batch spread can be compared against the tolerance bands that produced
the §10 WARN list.

Non-numeric leaves, booleans, and the ``metadata`` / ``*_per_channel`` blocks
are skipped (they are provenance/per-asset detail, not headline metrics).

Outputs:
    <root>/<ablation>/batch_metrics_aggregate.json
    <root>/<ablation>/batch_metrics_aggregate.csv

Usage:
    python scripts/mini_exp/aggregate_batch_metrics.py \
        --root /home/epham/StonkBench/outputs/generation \
        --ablation simulation_v2 fair_20261004 latest
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SKIP_PREFIXES = ("metadata",)
SKIP_SUFFIXES = ("_per_channel",)


def _flatten(obj: Any, prefix: str = "") -> Dict[str, float]:
    """Numeric leaves only, keyed by dotted path."""
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(path.startswith(p + ".") or path == p for p in SKIP_PREFIXES):
                continue
            if str(key).endswith(SKIP_SUFFIXES):
                continue
            out.update(_flatten(value, path))
    elif isinstance(obj, bool):
        return out
    elif isinstance(obj, (int, float)):
        if obj != obj:  # NaN
            return out
        out[prefix] = float(obj)
    return out


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _batch_summaries(ablation: str, root: Path, batch: int) -> List[Path]:
    """Per-cell summaries for one batch, falling back to a batch-level one."""
    batch_dir = root / ablation / f"evaluation_batch{batch}"
    summaries = [
        path
        for path in sorted(batch_dir.glob("*/complete_evaluation.json"))
        # ``shard_*`` directories are the superseded sharded layout (partial
        # summaries that overlap the per-cell ones); never mix the two.
        if not path.parent.name.startswith("shard_")
    ]
    if summaries:
        return summaries
    legacy = batch_dir / "complete_evaluation.json"
    return [legacy] if legacy.exists() else []


def aggregate(ablation: str, root: Path, batches: Iterable[int]) -> Tuple[Dict[str, Any], List[dict]]:
    per_cell: Dict[str, Dict[str, List[float]]] = {}
    sources: Dict[str, List[int]] = {}
    missing: List[int] = []
    for batch in batches:
        summaries = _batch_summaries(ablation, root, batch)
        if not summaries:
            print(f"[agg] {ablation}: no cell summaries in evaluation_batch{batch} (skipped)")
            missing.append(batch)
            continue
        for summary in summaries:
            data = _load(summary)
            for cell, payload in data.items():
                flat = _flatten(payload)
                if not flat:
                    print(f"[agg] {ablation}: empty metrics for {cell} in {summary}")
                bucket = per_cell.setdefault(cell, {})
                for metric, value in flat.items():
                    bucket.setdefault(metric, []).append(float(value))
                sources.setdefault(cell, []).append(batch)

    rows: List[dict] = []
    table: Dict[str, Any] = {
        "ablation": ablation,
        "batches_missing": missing,
        "cells": {},
    }
    for cell in sorted(per_cell):
        metrics_out: Dict[str, Any] = {}
        for metric, values in sorted(per_cell[cell].items()):
            stats = {
                "n_batches": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "values": values,
            }
            metrics_out[metric] = stats
            rows.append(
                {
                    "ablation": ablation,
                    "cell": cell,
                    "metric": metric,
                    **{k: stats[k] for k in ("n_batches", "mean", "std", "min", "max")},
                }
            )
        table["cells"][cell] = {
            "batches_present": sources.get(cell, []),
            "metrics": metrics_out,
        }
    return table, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="generation root")
    ap.add_argument("--ablation", nargs="+", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--headline", nargs="*", default=None,
                    help="metric path suffixes to print (default: a small set)")
    args = ap.parse_args()

    root = Path(args.root)
    headline = args.headline or [
        "FidelityEvaluator.kd",
        "FidelityEvaluator.cmd",
        "DiversityEvaluator.icd_euclidean",
        "StylizedFactsEvaluator",
    ]

    exit_code = 0
    for ablation in args.ablation:
        table, rows = aggregate(ablation, root, args.batches)
        if not rows:
            print(f"[agg] {ablation}: no evaluation summaries found — nothing to do")
            exit_code = 1
            continue

        out_json = root / ablation / "batch_metrics_aggregate.json"
        out_csv = root / ablation / "batch_metrics_aggregate.csv"
        out_json.write_text(json.dumps(table, indent=2, sort_keys=True, default=str))
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "ablation", "cell", "metric",
                    "n_batches", "mean", "std", "min", "max",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n[agg] {ablation}: {len(table['cells'])} cells, {len(rows)} metric series")
        if table["batches_missing"]:
            print(f"      [WARN] batches without summaries: {table['batches_missing']}")
        print(f"      -> {out_json}\n      -> {out_csv}")
        for cell, payload in sorted(table["cells"].items()):
            shown = []
            for metric, stats in payload["metrics"].items():
                if any(metric.endswith(h) or h in metric for h in headline):
                    shown.append(
                        f"{metric.split('.')[-1]}={stats['mean']:.4g}"
                        f"±{stats['std']:.3g}(n={stats['n_batches']})"
                    )
            if shown:
                print(f"      {cell}: " + "  ".join(shown))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
