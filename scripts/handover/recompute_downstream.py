#!/usr/bin/env python3
"""Recompute the downstream (`pnl`, `portfolio`) blocks with the fixed evaluators.

The shipped run used the pre-fix implementations in ``src/taxonomies/pnl.py`` and
``src/taxonomies/portfolio.py``; those defects are documented in
``handover/KNOWN_ISSUES.md``.  This script re-evaluates every cell with the
current (fixed) code and rewrites

    handover/04_evaluation_results/complete_evaluation.json

in place, leaving the fidelity / diversity / stylized-facts / hedging blocks
untouched (the fixes do not reach them).  A per-cell record of what changed is
written to

    handover/04_evaluation_results/evaluation_metadata.json

Only ``numpy`` and ``scipy`` are needed: the two evaluators are exercised on
their numpy code paths.  ``torch`` is stubbed out so the modules import.

    PYTHONPATH=<libs> python3 scripts/handover/recompute_downstream.py
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parents[1]
HANDOVER = REPO / "handover"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(REPO))


from torch_stub import install_torch_stub  # noqa: E402

# Only the numpy code paths of the evaluators run here, so a real torch install
# is not required; the shared stub is enough for ``src.taxonomies`` to import.
install_torch_stub()

from torch_pt_reader import load_pt  # noqa: E402
from src.taxonomies.pnl import PnLEvaluator  # noqa: E402
from src.taxonomies.portfolio import PortfolioOptimizationEvaluator  # noqa: E402

RESULTS = HANDOVER / "04_evaluation_results" / "complete_evaluation.json"
METADATA = HANDOVER / "04_evaluation_results" / "evaluation_metadata.json"
# The immutable as-shipped run.  It is the baseline the `ported_from_shipped`
# record is read from, so this script must never read its own previous output
# back in (that would make the record drift on every re-run).
SHIPPED = REPO / "outputs" / "results" / "latest" / "evaluation" / "complete_evaluation.json"
GENERATED = HANDOVER / "02_generated_outputs"
REFERENCE = HANDOVER / "03_ground_truth" / "reference"

# Metrics that the fix changed the meaning of, per taxonomy, so a reader knows
# which columns of the old run are not comparable with the new ones.
PNL_KEYS = (
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "omega_ratio",
    "tail_ratio",
    "annualized_return",
    "annualized_volatility",
    "terminal_wealth_mean",
    "terminal_wealth_std",
    "terminal_wealth_min",
    "terminal_wealth_max",
    "prob_positive_return",
)


def _generated(model: str, length: int) -> np.ndarray:
    return load_pt(GENERATED / model / f"{model}_seq{length}.pt")["data"].astype(np.float64)


def _reference(model: str, length: int) -> np.ndarray:
    with np.load(REFERENCE / f"{model}_seq{length}.npz") as z:
        return z["data"].astype(np.float64)


def _flatten_old(block: dict, arm: str) -> dict:
    return block.get(arm, {})


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    if not SHIPPED.exists():
        raise SystemExit(f"missing the as-shipped baseline {SHIPPED}")
    shipped = json.loads(SHIPPED.read_text())

    alignment = {}
    with (CODE_DIR / "data" / "evaluation_alignment.csv").open() as fh:
        for row in csv.DictReader(fh):
            alignment[row["cell"]] = row

    cells = {}
    for cell_id, cell in shipped.items():
        model, length_s = cell_id.rsplit("_seq", 1)
        length = int(length_s)

        real = _reference(model, length)
        synth_full = _generated(model, length)
        n = int(real.shape[0])
        if synth_full.shape[0] < n:
            raise SystemExit(f"{cell_id}: only {synth_full.shape[0]} generated samples for n={n}")
        synth = synth_full[:n]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pnl = PnLEvaluator(real_log_returns=real, synthetic_log_returns=synth).evaluate()
            portfolio = PortfolioOptimizationEvaluator(
                real_test_returns=real, synthetic_returns=synth
            ).evaluate()

        old_gmv = {
            na: np.asarray(strats["GMV_LongOnly"]["weights"], dtype=float)
            for na, strats in cell.get("portfolio", {}).items()
            if isinstance(strats, dict) and "GMV_LongOnly" in strats
        }
        correction = {
            "n_real_windows": n,
            "n_synthetic_used": int(synth.shape[0]),
            "reference_pool": alignment.get(cell_id, {}).get("reference_pool"),
            "reference_pool_size": alignment.get(cell_id, {}).get("reference_pool_size"),
            "recomputed": ["pnl", "portfolio"],
            "shipped_gmv_long_only_equalled_1N": {
                na: bool(np.allclose(w, 1.0 / len(w))) for na, w in old_gmv.items()
            },
            "ported_from_shipped": {
                arm: {
                    k: _flatten_old(cell.get("pnl", {}), arm).get(k)
                    for k in ("terminal_wealth_mean", "sortino_ratio", "calmar_ratio")
                }
                for arm in ("real", "synthetic")
            },
        }
        cell["pnl"] = pnl
        cell["portfolio"] = portfolio
        cell["downstream_recomputed_with_fixed_evaluators"] = correction
        cells[cell_id] = cell
        print(f"  {cell_id:<42} n={n:>4}  portfolio strategies={len(next(iter(portfolio.values())))}")

    RESULTS.write_text(json.dumps(cells, indent=1, sort_keys=True, allow_nan=True))

    metadata = {
        "generator": "scripts/handover/recompute_downstream.py",
        "results_file": "complete_evaluation.json",
        "blocks_recomputed": ["pnl", "portfolio"],
        "blocks_shipped_unchanged": ["FidelityEvaluator", "DiversityEvaluator",
                                     "StylizedFactsEvaluator", "utility"],
        "fixes_applied": [
            "log-return compounding uses exp(cumsum(r)) / exp(sum(r)) instead of cumprod(1+r) / prod(1+r),"
            " in both the P&L block and the portfolio block's drawdown path",
            "ratio metrics with an undefined denominator are reported as inf/nan and counted, not epsilon-patched to ~1e15",
            "long-only GMV/tangency use scaled, multi-start optimisers and warn instead of silently returning 1/N",
            "the four real-data baselines keep distinct keys (Real_GT_RealGMV[_LongOnly], Real_GT_RealTangency[_LongOnly])",
            "turnover is null when no weight history exists",
        ],
        "notes": [
            "complete_evaluation.json is Python-extended JSON: it contains NaN literals "
            "(20 of them, all in StylizedFactsEvaluator.long_memory_volatility.{real,diff} "
            "for 10 of the 48 cells). They are inherited unchanged from the shipped run. "
            "Use json.loads(..., parse_constant=...) if you need strict JSON.",
            "turnover is null rather than 0 when no rebalancing history exists.",
            "pnl.*.sortino_ratio can be null/inf where a path has no downside; "
            "n_sortino_defined / n_sortino_undefined record how many paths the mean used.",
        ],
        "alignment": list(alignment.values()),
        "metric_directions": _read_csv(CODE_DIR / "data" / "metric_directions.csv"),
        "cell_index": _read_csv(CODE_DIR / "data" / "cells.csv"),
    }
    METADATA.write_text(json.dumps(metadata, indent=1, sort_keys=True))
    print(f"\nwrote {RESULTS.relative_to(REPO)}")
    print(f"wrote {METADATA.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
