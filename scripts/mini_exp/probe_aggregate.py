#!/usr/bin/env python
"""Aggregate seed-sweep probe results for the WARN-resolution audit.

Reads ``sanity_outputs.json`` produced by ``score_outputs.py`` for the
original run (simulation_v2, seed 42) and the per-seed probe runs
(probe_s43 / probe_s44 / probe_s45, plus the salvaged probe_seed tree),
then prints one row per (model, seq) with every seed's std_ratio/verdict.

Interpretation guide (used by the audit):
  * seeds straddling the [0.8, 1.25] band  -> seed-noise cell (rerun-on-flag
    remediation is fair: the cell's expectation sits inside the band)
  * all seeds outside the band, same side  -> intrinsic model signature
    (document with evidence; no config was ever tuned)
  * PASS cells drifting across the band    -> baseline variance estimate
    (fairness control: shows band edges move for every model)

Usage:
    python scripts/mini_exp/probe_aggregate.py \
        --root /home/epham/StonkBench/outputs/mini_exp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# run_id -> seed label (simulation_v2 = the original seed-42 run)
RUNS = [
    ("simulation_v2", "42"),
    ("probe_s43", "43"),
    ("probe_s44", "44"),
    ("probe_s45", "45"),
    ("probe_seed", "4?"),  # salvaged partial run (mixed seeds, last-writer)
]


def load_rows(report_dir: Path) -> dict:
    p = report_dir / "sanity_outputs.json"
    if not p.exists():
        return {}
    rows = json.loads(p.read_text())
    return {(r["model"], r["seq_len"]): r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="default: /home/epham/StonkBench/outputs/mini_exp")
    ap.add_argument("--salvage", action="store_true",
                    help="include the salvaged probe_seed tree (mixed seeds)")
    args = ap.parse_args()

    root = Path(args.root or Path(__file__).resolve().parents[2] / "outputs" / "mini_exp")

    runs = RUNS if args.salvage else RUNS[:-1]
    data = {}  # (model, seq) -> {seed_label: (ratio, verdict)}
    for run_id, seed in runs:
        rows = load_rows(root / run_id)
        for k, r in rows.items():
            data.setdefault(k, {})[seed] = (
                r.get("std_ratio_mean"), r.get("verdict"), r.get("flags", []),
            )

    print(f"{'model':28s} {'seq':>4s} | " + " ".join(f"{s:>16s}" for _, s in runs) + " | verdict-set")
    print("-" * 130)
    for k in sorted(data):
        cells = data[k]
        parts, verdicts = [], set()
        for _, seed in runs:
            if seed in cells:
                ratio, v, flags = cells[seed]
                parts.append(f"{ratio:6.3f} {v:<4s}".ljust(16))
                verdicts.add(v)
            else:
                parts.append("-".ljust(16))
        print(f"{k[0]:28s} {k[1]:>4d} | " + " ".join(parts) + " | " + ",".join(sorted(verdicts)))

    # compact PASS/WARN census across probe seeds only (43/44/45)
    for _, seed in runs[1:]:
        n_pass = sum(1 for k, c in data.items() if seed in c and c[seed][1] == "PASS")
        n_tot = sum(1 for k, c in data.items() if seed in c)
        print(f"seed {seed}: PASS={n_pass}/{n_tot}")


if __name__ == "__main__":
    main()
