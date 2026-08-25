#!/usr/bin/env python
"""Repair utility summaries in evaluation metrics.json files.

The pre-2026-08-07 evaluator aggregation flattened only scalar leaf values,
so hedgers whose payloads are nested dicts (``{'real_train': {...},
'mixed_train': {...}}``) produced empty summaries. The per-channel data was
always correct; this recomputes ``utility.summary`` from ``utility.per_channel``
using recursive nested averaging.

Usage:
    python scripts/repair_utility_summaries.py --results_dir <evaluation_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def average_nested_dicts(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively average a list of (possibly nested) metric dicts."""
    if not results:
        return {}
    first = results[0]
    averaged: Dict[str, Any] = {}
    for key, value in first.items():
        values = [r[key] for r in results if key in r]
        if not values:
            continue
        if isinstance(value, dict):
            averaged[key] = average_nested_dicts(values)
        elif isinstance(value, (int, float)):  # includes numpy floats via JSON load
            averaged[key] = float(sum(float(v) for v in values) / len(values))
        else:
            averaged[key] = value
    return averaged


def repair_metrics(metrics_path: Path) -> bool:
    """Rewrite ``utility.summary`` from ``utility.per_channel`` in place.

    ``utility`` has the shape ``{'summary': ..., 'per_channel': [{'augmented_testing': {...}}, ...]}``:
    the aggregation keys (e.g. ``augmented_testing``) live *inside* each
    per-channel result, not at the utility top level. The summary is rebuilt
    by averaging each such per-channel key across all channels.
    """
    try:
        with metrics_path.open() as f:
            data = json.load(f)
    except Exception:
        return False

    utility = data.get("utility")
    if not isinstance(utility, dict):
        return False
    per_channel = utility.get("per_channel")
    if not isinstance(per_channel, list) or not per_channel:
        return False

    # Find the per-channel metric keys (e.g. "augmented_testing") from the
    # channel payloads themselves, skipping non-dict values (errors).
    channel_keys: set = set()
    for r in per_channel:
        if isinstance(r, dict):
            channel_keys.update(
                k for k, v in r.items()
                if k not in ("summary", "per_channel") and isinstance(v, dict)
            )

    aggregated: Dict[str, Any] = {}
    for key in sorted(channel_keys):
        channel_values = [r.get(key, {}) for r in per_channel if isinstance(r, dict)]
        all_hedgers = set()
        for cv in channel_values:
            if isinstance(cv, dict):
                all_hedgers.update(k for k, v in cv.items() if isinstance(v, dict))
        if not all_hedgers:
            continue
        aggregated[key] = {
            hedger: average_nested_dicts(
                [cv[hedger] for cv in channel_values
                 if isinstance(cv, dict) and hedger in cv and isinstance(cv[hedger], dict)]
            )
            for hedger in sorted(all_hedgers)
        }

    # Never clobber an already-good summary with an empty one.
    if not aggregated and not utility.get("summary"):
        return True

    utility["summary"] = aggregated
    with metrics_path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True, type=Path,
                        help="Evaluation directory containing seq_*/*/metrics.json")
    args = parser.parse_args()

    fixed = 0
    empty = 0
    for metrics_path in sorted(args.results_dir.glob("seq_*/*/metrics.json")):
        if repair_metrics(metrics_path):
            fixed += 1
        else:
            empty += 1
    print(f"Repaired {fixed} metrics.json summaries ({empty} skipped/empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
