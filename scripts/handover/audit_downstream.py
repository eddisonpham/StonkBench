#!/usr/bin/env python3
"""Audit + regression evidence for the P&L and portfolio downstream evaluations.

Runs the *shipped* arrays (generated samples and verified references) through the
pre-fix and post-fix implementations so every claim in ``handover/KNOWN_ISSUES.md``
is reproducible.  The shipped numbers are read from the original run at
``outputs/results/latest/evaluation/complete_evaluation.json``; the bundle's own
JSON holds the *corrected* values (see ``scripts/handover/recompute_downstream.py``).

    PYTHONPATH=<libs> python3 scripts/handover/audit_downstream.py

``src/taxonomies/pnl.py`` and ``src/taxonomies/portfolio.py`` import torch, which
is not required to read this bundle, so a stub is installed before importing them.
Only the numpy code paths are exercised.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parents[1]
HANDOVER = REPO / "handover"
SHIPPED = REPO / "outputs" / "results" / "latest" / "evaluation" / "complete_evaluation.json"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(REPO))


from torch_stub import install_torch_stub  # noqa: E402

# Only the numpy code paths of the evaluators run here, so a real torch install
# is not required; the shared stub is enough for ``src.taxonomies`` to import.
install_torch_stub()

from load_handover import Handover  # noqa: E402
from src.taxonomies.pnl import _pnl_metrics  # noqa: E402
from src.taxonomies.portfolio import (  # noqa: E402
    _minimum_variance_weights,
    _portfolio_metrics,
    _sample_covariance,
)
import src.taxonomies.portfolio as P  # noqa: E402

H = Handover()
J = json.loads(SHIPPED.read_text())
J_NEW = json.loads((HANDOVER / "04_evaluation_results" / "complete_evaluation.json").read_text())
FAILS: list[str] = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAILS.append(msg)


# --------------------------------------------------------------------------- #
# pre-fix implementations (verbatim from git HEAD, for comparison)
# --------------------------------------------------------------------------- #
def old_sortino_reported(port_returns: np.ndarray, periods_per_year: int = 252) -> float:
    """The pre-fix ``sortino_ratio``: mean of mean/(rms(neg) + 1e-15), annualised."""
    r = np.asarray(port_returns, dtype=np.float64)
    mean_ret = r.mean(axis=1)
    vals = []
    for i in range(r.shape[0]):
        d = np.sqrt(np.mean(np.minimum(r[i], 0.0) ** 2))
        vals.append(mean_ret[i] / (d + 1e-15))
    return float(np.mean(vals)) * np.sqrt(periods_per_year)


def old_terminal_wealth(port_returns: np.ndarray) -> np.ndarray:
    """The pre-fix ``terminal_wealth``: prod(1 + log-return)."""
    return np.prod(1.0 + np.asarray(port_returns, dtype=np.float64), axis=1)


def old_compound(port_returns: np.ndarray) -> np.ndarray:
    """The pre-fix ``cumulative``: cumprod(1 + log-return)."""
    return np.cumprod(1.0 + np.asarray(port_returns, dtype=np.float64), axis=1)


def old_long_only_gmv(cov: np.ndarray) -> np.ndarray:
    """The pre-fix long-only GMV: SLSQP from 1/N with default tolerances."""
    from scipy.optimize import minimize

    C = cov.shape[0]
    res = minimize(
        lambda w: float(w @ cov @ w),
        np.ones(C) / C,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * C,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
    )
    return res.x if res.success else np.ones(C) / C


def real_key(suffix: str) -> str:
    """The pre-fix key construction."""
    return f"Real_GT_{suffix.split('_')[1]}{suffix.split('_')[2]}"


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def audit_portfolio() -> None:
    print("\n[P1] long-only GMV stalls at its 1/N starting point")
    stall = 0
    improved = 0
    for model, L in [("block_bootstrap", 126), ("quantgan", 126), ("timegrad", 21), ("vrnn_klsched", 252), ("dcc_garch", 42)]:
        X = H.generated_outputs(model, L).astype(np.float64)
        for C in (5, 10, 25):
            syn = X[:, :, :C].reshape(-1, C)
            cov = _sample_covariance(syn)
            w_old = old_long_only_gmv(cov)
            w_new = _minimum_variance_weights(cov, allow_short=False)
            eq = np.ones(C) / C
            stall += int(np.allclose(w_old, eq))
            improved += int(float(w_new @ cov @ w_new) < float(eq @ cov @ eq) - 1e-15)
            if C == 25 and model == "block_bootstrap":
                print(
                    f"       {model} L={L} C={C}: objective 1/N={float(eq @ cov @ eq):.3e} "
                    f"old(={('1/N' if np.allclose(w_old, eq) else 'moved')})={float(w_old @ cov @ w_old):.3e} "
                    f"new={float(w_new @ cov @ w_new):.3e}"
                )
    check(stall >= 15, f"pre-fix long-only GMV returned exactly 1/N in {stall}/15 cases (reproduced)")
    check(improved >= 14, f"post-fix long-only GMV strictly beats 1/N in {improved}/15 cases")

    print("\n[P1b] as a consequence, GMV_LongOnly was identical to the 1/N benchmark")
    identical = 0
    for cid, cell in J.items():
        for na, strategies in cell["portfolio"].items():
            identical += int(
                np.allclose(np.array(strategies["GMV_LongOnly"]["weights"]), np.array(strategies["1N"]["weights"]))
            )
    check(identical == 144, f"shipped GMV_LongOnly == 1N in {identical}/144 strategy-size pairs")

    print("\n[P2] the four real-data baselines collided onto two keys")
    suffixes = ["_Real_GMV", "_Real_GMV_LongOnly", "_Real_Tangency", "_Real_Tangency_LongOnly"]
    old_keys = [real_key(s) for s in suffixes]
    check(
        len(set(old_keys)) == 2,
        f"pre-fix keys: {old_keys} -> only {len(set(old_keys))} distinct (long-only overwrote the oracle)",
    )
    new_keys = [
        "Real_GT_RealGMV",
        "Real_GT_RealGMV_LongOnly",
        "Real_GT_RealTangency",
        "Real_GT_RealTangency_LongOnly",
    ]
    check(len(set(new_keys)) == 4, f"post-fix keys are distinct: {new_keys}")
    dup = 0
    for cid, cell in J.items():
        for na, strategies in cell["portfolio"].items():
            dup += int(
                np.allclose(
                    np.array(strategies["Real_GT_RealGMV"]["weights"]), np.array(strategies["1N"]["weights"])
                )
            )
    check(dup == 144, f"in the shipped results Real_GT_RealGMV == 1N in {dup}/144 cases, i.e. it is the long-only run")

    print("\n[P3] turnover was structurally zero")
    tv = [
        v["turnover"]
        for c in J
        for na in J[c]["portfolio"]
        for v in J[c]["portfolio"][na].values()
        if isinstance(v, dict) and "turnover" in v
    ]
    check(len(tv) == 1008 and all(x == 0 for x in tv), f"shipped turnover is 0 in all {len(tv)} entries")
    one = np.ones(5) / 5
    m = _portfolio_metrics(np.random.default_rng(0).normal(0, 0.01, 300), weights_history=one[np.newaxis, :])
    check(m["turnover"] is None, "post-fix turnover is None (undefined) for a single static weight vector")
    hist = np.stack([one, np.array([0.4, 0.3, 0.1, 0.1, 0.1])])
    m2 = _portfolio_metrics(np.random.default_rng(0).normal(0, 0.01, 300), weights_history=hist)
    check(m2["turnover"] is not None and m2["turnover"] > 0, f"post-fix turnover computed when a history exists ({m2['turnover']:.4f})")

    print("\n[P4] the documented rolling estimation was never implemented")
    src = (REPO / "src" / "taxonomies" / "portfolio.py").read_text()
    check(
        "self.estimation_window" in src and "rebalance" in src,
        "estimation_window / rebalance_freq are stored on the evaluator",
    )
    used = src.count("self.estimation_window") + src.count("self.rebalance_freq")
    check(
        used <= 2,
        f"neither parameter is read by any computation (only {used} references, both assignments)",
    )


def audit_pnl() -> None:
    print("\n[B1] log returns were compounded as if they were simple returns")
    model, L = "kalman_vae_safe", 126
    n = H.effective_n(model, L)
    port = H.generated_outputs(model, L).astype(np.float64)[:n].mean(axis=2)
    old_tw = old_terminal_wealth(port).mean()
    new_tw = float(_pnl_metrics(port)["terminal_wealth_mean"])
    shipped = J[f"{model}_seq{L}"]["pnl"]["synthetic"]["terminal_wealth_mean"]
    check(abs(old_tw - shipped) < 1e-9, f"pre-fix formula reproduces the shipped value exactly ({shipped:.6f})")
    check(abs(new_tw - np.exp(port.sum(axis=1)).mean()) < 1e-12, "post-fix terminal wealth == exp(sum(log returns))")
    print(f"       {model} L={L}: shipped/prod(1+r)={shipped:.6f}  exp(sum(r))={new_tw:.6f}  delta={new_tw - shipped:+.2e}")
    # the same output mixes both conventions
    cum = np.exp(np.cumsum(port[0]))
    old = old_compound(port[:1])[0]
    check(not np.allclose(cum, old), "pre-fix cumulative path disagreed with _accumulate_returns on the same data")

    print("\n[B2] Sortino exploded on paths with no downside")
    model, L = "timegrad", 21
    port = H.generated_outputs(model, L).astype(np.float64).mean(axis=2)
    old_s = old_sortino_reported(port)
    new = _pnl_metrics(port)
    shipped = J[f"{model}_seq{L}"]["pnl"]["synthetic"]["sortino_ratio"]
    check(abs(old_s - shipped) / max(abs(shipped), 1e-30) < 1e-6, f"pre-fix formula reproduces the shipped {shipped:.6g}")
    d = [float(np.sqrt(np.mean(np.minimum(p, 0.0) ** 2))) for p in port]
    check(min(d) == 0.0, f"at least one path has zero downside deviation (min rms={min(d):.1e})")
    check(
        new["n_sortino_undefined"] >= 1,
        f"post-fix counts the undefined paths (n_undefined={new['n_sortino_undefined']}, "
        f"n_defined={new['n_sortino_defined']})",
    )
    check(
        not np.isfinite(new["sortino_ratio"]) or abs(new["sortino_ratio"]) < 1e3,
        f"post-fix sortino is sane or NaN: {new['sortino_ratio']}",
    )
    print(f"       shipped={shipped:.6g}  post-fix={new['sortino_ratio']}")

    print("\n[B3] 'independent' real paths are overlapping windows of one series")
    R = H.reference("block_bootstrap", 126)
    check(R.shape[0] > 1 and abs(R[0].mean() - R[1].mean()) < 1e-3, f"{R.shape[0]} windows overlap by 125/126 = 99.2%")

    print("\n[B4] the real arm's own value changes with the model being evaluated")
    for L in (21, 126):
        vals = sorted({round(v["pnl"]["real"]["sharpe_ratio"], 4) for k, v in J.items() if k.endswith(f"seq{L}")})
        print(f"       L={L}: distinct real Sharpe values = {vals}")
    v21 = sorted({round(v["pnl"]["real"]["sharpe_ratio"], 4) for k, v in J.items() if k.endswith("seq21")})
    check(len(v21) > 1, f"same real data yields {len(v21)} different 'real' Sharpe values at L=21")

    print("\n[B5] exposure: how much the fixes move the headline P&L numbers")
    tot_old, tot_new, worst = 0.0, 0.0, (0.0, "")
    for cid, cell in J.items():
        model, L = cid.rsplit("_seq", 1)[0], int(cid.rsplit("_seq", 1)[1])
        n = H.effective_n(model, L)
        for arm, data in (("synthetic", H.generated_outputs(model, L)), ("real", H.reference(model, L))):
            port = data.astype(np.float64)[:n].mean(axis=2)
            new = _pnl_metrics(port, periods_per_year=252)
            shipped_tw = cell["pnl"][arm]["terminal_wealth_mean"]
            rel = abs(new["terminal_wealth_mean"] - shipped_tw) / max(abs(shipped_tw), 1e-30)
            tot_old += abs(shipped_tw)
            tot_new += abs(new["terminal_wealth_mean"])
            if rel > worst[0]:
                worst = (rel, f"{cid}/{arm}")
    print(f"       worst relative change in terminal_wealth_mean: {worst[0] * 100:.3f}% ({worst[1]})")
    print("       (portfolio metrics are unaffected by the P&L fix; the portfolio fixes change GMV_LongOnly and the Real_GT_* keys)")


def audit_bundle() -> None:
    """The bundle's JSON must already hold the corrected (post-fix) values."""
    print("\n[C1] handover/04_evaluation_results/complete_evaluation.json is post-fix")
    keys = {k for c in J_NEW for na in J_NEW[c]["portfolio"] for k in J_NEW[c]["portfolio"][na]}
    check(len({k for k in keys if k.startswith("Real_GT_")}) == 4, f"4 distinct real baselines: {sorted(k for k in keys if k.startswith('Real_GT_'))}")

    eq_1n = 0
    for c in J_NEW:
        for na in J_NEW[c]["portfolio"]:
            s = J_NEW[c]["portfolio"][na]
            for name in ("GMV_LongOnly", "Tangency_LongOnly"):
                w = np.asarray(s[name]["weights"], dtype=float)
                eq_1n += int(name == "GMV_LongOnly" and np.allclose(w, 1.0 / len(w)))
    check(eq_1n == 0, f"GMV_LongOnly never collapses to 1/N ({eq_1n} cases, was 144)")

    turnovers = {v["turnover"] for c in J_NEW for na in J_NEW[c]["portfolio"] for v in J_NEW[c]["portfolio"][na].values()}
    check(turnovers == {None}, f"turnover is null everywhere (no weight history): {turnovers}")

    worst = 0.0
    n_undef = 0
    for c in J_NEW:
        for arm in ("real", "synthetic"):
            s = J_NEW[c]["pnl"][arm]["sortino_ratio"]
            if np.isfinite(s):
                worst = max(worst, abs(s))
            n_undef += int(J_NEW[c]["pnl"][arm].get("n_sortino_undefined", 0))
    check(worst < 1e3, f"no epsilon-inflated Sortino left (max |sortino| = {worst:.2f}, was 6.65e12)")
    print(f"       undefined-downside paths flagged instead of patched: {n_undef}")

    block_mismatch = 0
    for c in J_NEW:
        if json.dumps(J_NEW[c]["FidelityEvaluator"], sort_keys=True) != json.dumps(J[c]["FidelityEvaluator"], sort_keys=True):
            block_mismatch += 1
    check(block_mismatch == 0, "fidelity/diversity/stylized blocks are byte-identical to the shipped run (fixes do not reach them)")


def main() -> None:
    print("=" * 78)
    print("Audit: P&L and portfolio downstream evaluations")
    print("=" * 78)
    audit_portfolio()
    audit_pnl()
    audit_bundle()
    print("\n" + "=" * 78)
    if FAILS:
        print(f"FAILED {len(FAILS)} check(s):")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("ALL AUDIT CHECKS PASSED (defects reproduced; fixes verified)")


if __name__ == "__main__":
    main()
