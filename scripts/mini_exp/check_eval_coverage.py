#!/usr/bin/env python
"""Report which of the 432 evaluation cells are complete.

A cell counts as complete only when

    <root>/<ablation>/evaluation_batch<k>/<model>_seq<L>/complete_evaluation.json

exists and holds exactly one non-empty entry keyed by that directory name (the
same rule the eval job applies before it declares itself done). Optional
downstream blocks are reported too, so a cell that finished but silently
skipped, say, ``portfolio`` is visible rather than averaged in.

Usage:
    python scripts/mini_exp/check_eval_coverage.py --root <generation root> \
        [--ablation simulation_v2 fair_20261004 latest] [--quiet]

Exit codes: 0 = every expected cell complete, 1 = cells missing or invalid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ABLATIONS = ["simulation_v2", "fair_20261004", "latest"]
MODELS = [
    "block_bootstrap", "conditional_tsdiffusion", "cond_sig_wgan",
    "de_jump_diffusion", "dcc_garch", "kalman_vae_safe", "merton_jump_diffusion",
    "quantgan", "stationary_block_bootstrap", "timegrad",
    "unconditional_tsdiffusion", "vrnn_klsched",
]
SEQS = [21, 42, 126, 252]
BATCHES = [1, 2, 3]

# Blocks the evaluator is expected to produce for every cell.
REQUIRED_BLOCKS = (
    "FidelityEvaluator", "DiversityEvaluator", "StylizedFactsEvaluator",
    "utility", "portfolio", "pnl",
)


def expected_cells(ablations):
    return {
        (abl, batch, model, seq)
        for abl in ablations
        for batch in BATCHES
        for model in MODELS
        for seq in SEQS
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="generation root")
    ap.add_argument("--ablation", nargs="+", default=ABLATIONS)
    ap.add_argument("--quiet", action="store_true", help="only print problems")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[coverage] root missing: {root}")
        return 2

    expected = expected_cells(args.ablation)
    found: dict = {}
    broken = []
    for path in root.glob("*/evaluation_batch*/*_seq*/complete_evaluation.json"):
        parts = path.parent.relative_to(root).parts
        if len(parts) != 3:
            continue
        abl, batch_dir, cell_dir = parts
        if abl not in args.ablation:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # truncated / unreadable
            broken.append((str(path), f"unreadable: {exc}"))
            continue
        if list(data) != [cell_dir] or not data[cell_dir]:
            broken.append((str(path), f"keys={list(data)}"))
            continue
        model, seq = cell_dir.rsplit("_seq", 1)
        found[(abl, int(batch_dir.replace("evaluation_batch", "")), model, int(seq))] = data[cell_dir]

    missing = sorted(expected - set(found))
    unexpected = sorted(set(found) - expected)

    per_ablation = {}
    for abl in args.ablation:
        done = sum(1 for key in found if key[0] == abl)
        per_ablation[abl] = f"{done}/{len(expected) // len(args.ablation)}"

    print(f"[coverage] {len(found)}/{len(expected)} cells complete  " +
          "  ".join(f"{k}={v}" for k, v in per_ablation.items()))

    partial = {
        key: sorted(set(REQUIRED_BLOCKS) - set(payload))
        for key, payload in found.items()
        if set(REQUIRED_BLOCKS) - set(payload)
    }
    if partial and not args.quiet:
        print(f"[coverage] {len(partial)} cell(s) missing expected blocks:")
        for key, blocks in sorted(partial.items())[:10]:
            print(f"    {key}: missing {blocks}")
    if broken:
        print(f"[coverage] {len(broken)} unreadable/invalid summary file(s):")
        for path, why in broken[:10]:
            print(f"    {path}: {why}")
    if missing:
        print(f"[coverage] {len(missing)} cell(s) MISSING:")
        for key in missing[:20]:
            print(f"    {key}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
    if unexpected:
        print(f"[coverage] {len(unexpected)} unexpected cell(s):")
        for key in unexpected[:10]:
            print(f"    {key}")

    ok = not missing and not broken and not unexpected
    print("[coverage] OK — every expected cell is present and well-formed"
          if ok else "[coverage] INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
