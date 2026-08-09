"""Reproduce the augmented cells from notebooks 02/03 to find where NaN
leaks in. This mirrors the EXACT code in those notebooks so I can see
what the user is seeing interactively."""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/uyenlamho/Documents/vscode/CSCD94F25/Unified-benchmark-for-SDGFTS")

from src.utility import (
    MetricToolbox,
    PortfolioTask,
    AlphaTask,
    UtilityEvaluator,
)
from src.utility.metrics import METRIC_KEYS


def make(n, L, C, mu, sigma, seed):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.normal(loc=mu, scale=sigma, size=(n, L, C)).astype(np.float32))


def show(label, agg_or_dict):
    if isinstance(agg_or_dict, dict) and "mean" in agg_or_dict and "std" in agg_or_dict:
        # Single metric aggregate
        return f"{agg_or_dict['mean']:+.3f}±{agg_or_dict['std']:.3f}"
    if isinstance(agg_or_dict, dict) and all(
        isinstance(v, dict) and "mean" in v for v in agg_or_dict.values()
    ):
        return "[" + ", ".join(
            f"{k}={v['mean']:+.3f}±{v['std']:.3f}"
            for k, v in agg_or_dict.items()
        ) + "]"
    return str(type(agg_or_dict))


# =================== notebook 02 augmented cell ========================= #
print("\n========== NOTEBOOK 02 (Portfolio) augmented cell ==========\n")
real_train = make(64, 252, 25, 2e-4, 0.015, 11)
real_test  = make(32, 252, 25, 2e-4, 0.015, 12)
syn_train  = make(64, 252, 25, 5e-4, 0.020, 22)

task = PortfolioTask(num_inventory=20, num_hedge=5, seq_length=252, pnl_returns="simple")
ev = UtilityEvaluator(task=task, protocol="augmented", num_epochs=2, batch_size=16)
aug_result = ev.run(real_train, real_test, syn_train)
print(f"top-level keys: {sorted(aug_result.keys())}")
print(f"aug_result['real'] type: {type(aug_result['real'])}")
print(f"aug_result['real'] keys: {sorted(aug_result['real'].keys()) if isinstance(aug_result['real'], dict) else '(not dict)'}")

# This is the EXACT helper from notebook 02 (post-patch):
def agg_notebook02(payload):
    """Drill through ``AugmentedProtocol.run()`` shapes: per-channel
    tasks give ``{per_channel, extras}`` → ``aggregate_per_channel``;
    whole-window tasks give ``{aggregate, extras}`` → use ``aggregate``
    directly. Without this branching, ``MetricToolbox.aggregate`` sees
    an empty filter and emits all-NaN values."""
    if "per_channel" in payload:
        return MetricToolbox.aggregate_per_channel(payload["per_channel"])
    if "aggregate" in payload:
        return payload["aggregate"]
    return MetricToolbox.aggregate([payload])

print("\nnotebook-02 augmented cell output (PATCHED helper):")
for label, key in [("Real (M_r)", "real"),
                   ("Synthetic (M̂_g)", "synthetic"),
                   ("Augmented (M̃, full union)", "augmented")]:
    try:
        ag = agg_notebook02(aug_result[key])
        print(f"  {label:<32}: {show('ag', ag)}")
        for k in METRIC_KEYS:
            v = ag.get(k, {}).get("mean", "MISSING")
            nan_flag = "  NaN!" if isinstance(v, float) and np.isnan(v) else ""
            print(f"     {k:<14}: {v}{nan_flag}")
    except Exception as exc:
        print(f"  {label}: ERROR {exc}")


# =================== notebook 03 augmented cell ========================= #
print("\n========== NOTEBOOK 03 (Alpha) augmented cell ==========\n")
real_train_a = make(64, 252, 25, 3e-4, 0.020, 51)
real_test_a  = make(32, 252, 25, 3e-4, 0.020, 52)
syn_train_a  = make(64, 252, 25, 6e-4, 0.025, 52)

task_a = AlphaTask(num_assets=25, seq_length=252, long_only=True, pnl_returns="simple")
ev_a = UtilityEvaluator(task=task_a, protocol="augmented", num_epochs=2, batch_size=16)
aug_result_a = ev_a.run(real_train_a, real_test_a, syn_train_a)
print(f"top-level keys: {sorted(aug_result_a.keys())}")
print(f"aug_result_a['real'] type: {type(aug_result_a['real'])}")
print(f"aug_result_a['real'] keys: {sorted(aug_result_a['real'].keys()) if isinstance(aug_result_a['real'], dict) else '(not dict)'}")

print("\nnotebook-03 augmented cell output (PATCHED helper):")
def _topagg_notebook03(payload):
    """Drill through AugmentedProtocol shapes: per-channel → aggregate_per_channel;
    whole-window → use aggregate directly."""
    if "per_channel" in payload:
        return MetricToolbox.aggregate_per_channel(payload["per_channel"])
    if "aggregate" in payload:
        return payload["aggregate"]
    return MetricToolbox.aggregate([payload])

for label, key in [("Real (M_r)", "real"),
                   ("Synthetic (M̂_g)", "synthetic"),
                   ("Augmented (M̃, full union)", "augmented")]:
    try:
        ag = _topagg_notebook03(aug_result_a[key])
        print(f"  {label:<32}: {show(label, ag)}")
        for k in METRIC_KEYS:
            v = ag.get(k, {}).get("mean", "MISSING")
            nan_flag = "  NaN!" if isinstance(v, float) and np.isnan(v) else ""
            print(f"     {k:<14}: {v}{nan_flag}")
    except Exception as exc:
        print(f"  {label}: ERROR {exc}")
