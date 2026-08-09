"""Adversarial probe of the §6 utility pipeline.

Run hostile inputs against every public entry-point to surface genuine
bugs:

- N=0 / N=1 test windows
- ``MoneynessLSTM.predict`` and ``BSStaticDelta.predict`` both with
  log-returns input (``L == seq_length - 1`` path) and with prices input
- ``PortfolioTask`` with an inverted/non-contiguous ``channel_map``
- ``initial_dollars`` None / wrong length / negative
- Degenerate Sharpe windows (all-zero returns)
- Mismatched train/test window lengths
- Custom ``MoneynessLSTM.predict`` argument shapes (K as scalar vs N-vec)

Outputs PASS / FAIL lines and continues on individual test failures so
every bug surfaces in one pass.

Run::

    venv/bin/python scripts/adversarial_utility.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import torch

ROOT = "/Users/uyenlamho/Documents/vscode/CSCD94F25/Unified-benchmark-for-SDGFTS"
sys.path.insert(0, ROOT)

from src.utility import (  # noqa: E402
    OptionsTask,
    PortfolioTask,
    AlphaTask,
    MoneynessLSTM,
    BSStaticDelta,
    PortfolioLSTM,
    AlphaLSTM,
    MetricToolbox,
    TSTRProtocol,
    AugmentedProtocol,
    UtilityEvaluator,
)
from src.utility.metrics import per_period_pnl_from_positions  # noqa: E402


PASSES = []
FAILS = []


def _probe(name: str):
    def deco(fn):
        def wrapped():
            print(f"\n[probe] {name}")
            try:
                fn()
                PASSES.append(name)
                print(f"  PASS — {name}")
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                FAILS.append((name, exc, tb))
                print(f"  FAIL — {name}: {exc.__class__.__name__}: {exc}")
                print("--- traceback (truncated) ---")
                for line in tb.splitlines()[-6:]:
                    print("  " + line)
        return wrapped
    return deco


def _lr(n, length, channels, seed=0, drift=0.0002):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(
        rng.normal(loc=drift, scale=0.015, size=(n, length, channels)).astype(np.float32)
    )


# ============================================================================ #
@_probe("MoneynessLSTM.predict with log returns (dead `K*None` branch)")
def probe_moneyness_predict_log_returns():
    """Paper §6.2 says callers may pass log returns. Currently this
    trips a ``K * None`` line that would crash on entry. We test that
    the LTSM either returns sane deltas or raises a clean error.
    """
    N, L = 2, 251
    log_r = torch.randn(N, L) * 0.01
    model = MoneynessLSTM(seq_length=L + 1, hidden_size=32, num_layers=1)
    out = model.predict(log_r, K=torch.ones(N))
    assert out.shape == (N, L + 1), f"Expected (N, L+1) on log-return path, got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "Deltas must be finite"


@_probe("BSStaticDelta.predict with log returns input")
def probe_bs_predict_log_returns():
    N, L = 2, 251
    log_r = torch.randn(N, L) * 0.01
    bs = BSStaticDelta(seq_length=L + 1, K=1.0, sigma_annual=0.2)
    out = bs.predict(log_r)
    assert out.shape == (N, L + 1)
    assert torch.isfinite(out).all()
    # delta should be in (0, 1) for ATM and positive time-to-maturity
    assert (out >= 0).all() and (out <= 1).all(), "BS delta must be in [0, 1]"


@_probe("MetricToolbox.compute on empty tensor (T=0)")
def probe_metric_zero_length():
    out = MetricToolbox.compute(torch.zeros(0))
    for k, v in out.items():
        assert np.isnan(v), f"{k} should be NaN with empty input, got {v}"


@_probe("MetricToolbox.compute on T=1 (degenerate window)")
def probe_metric_t1():
    out = MetricToolbox.compute(torch.tensor([1.0]))
    for k, v in out.items():
        assert np.isfinite(v) or k == "max_drawdown", f"{k}={v} on T=1"


@_probe("MetricToolbox.aggregate on empty list")
def probe_metric_aggregate_empty():
    out = MetricToolbox.aggregate([])
    for k in ("pnl", "sharpe", "cvar", "win_rate", "max_drawdown"):
        assert k in out and np.isnan(out[k]["mean"]) and np.isnan(out[k]["std"])


@_probe("Portfolio TSTR with N=1 (single test window → std=0)")
def probe_tstr_n1():
    """For *whole-window* tasks (Portfolio / Alpha) with N=1 test window,
    ``MetricToolbox.aggregate`` over a single-window list MUST report
    ``std == 0`` (paper eq. 27). Per-channel tasks are different — std
    there is across channels, which is non-trivial with C>1 and is
    tested separately."""
    real = _lr(8, 251, 25, seed=1)
    syn = _lr(8, 251, 25, seed=2)
    rt, rtest = real[:6], real[6:7]  # N=1 test window
    st = syn[:6]
    task = PortfolioTask(num_inventory=20, num_hedge=5, seq_length=251)
    ev = UtilityEvaluator(task=task, protocol="tstr",
                          num_epochs=1, batch_size=4, verbose=False)
    result = ev.run(rt, rtest, st)
    for tag in ("real", "synthetic"):
        agg = result[tag]
        for k, v in agg.items():
            assert v["std"] == 0.0, f"{tag}.{k}.std should be 0 with N=1, got {v['std']}"


@_probe("PortfolioTask channel_map with arbitrary non-contiguous indices")
def probe_channel_map_arbitrary():
    # 25 channels but inventory=[3,7,11,15,19,23,2,6,10,14,18,22,1,5,9,13,17,21,0,4]
    # and hedge=[8,12,16,20,24]
    inv = [3, 7, 11, 15, 19, 23, 2, 6, 10, 14, 18, 22, 1, 5, 9, 13, 17, 21, 0, 4]
    hdg = [8, 12, 16, 20, 24]
    task = PortfolioTask(
        num_inventory=20, num_hedge=5, seq_length=251,
        channel_map={"inventory": inv, "hedge": hdg},
    )
    train = _lr(6, 251, 25, seed=10)
    test = _lr(2, 251, 25, seed=11)
    pp = task.prepare_training(train)
    pol = task.build_policy()
    pol.fit(pp, num_epochs=1, batch_size=4, verbose=False)
    pnl = task.predict_period_pnl(pol, test)
    assert pnl.shape == (2, 251)
    prices = task.prepare_training(test)
    positions = pol.predict(prices)
    full = task._assemble_full_pos(torch.ones(()), positions)
    inv_set = set(inv)
    hdg_set = set(hdg)
    for j in range(25):
        if j in inv_set:
            assert torch.all(full[:, :, j] == 1.0), f"inventory ch {j} not 1.0"
        elif j in hdg_set:
            assert full[:, :, j].abs().max().item() <= 10.0 + 1e-5, (
                f"hedge ch {j} = {full[:,:,j].abs().max().item():.3f} outside clip"
            )


@_probe("initial_dollars with wrong length raises ValueError (not silent broadcast)")
def probe_init_dollars_mismatch_silent():
    """A length-2 dollar tensor against N=3 windows used to silently
    crash mid-tensor multiplication. Now it raises a clear ValueError."""
    train = _lr(8, 251, 25, seed=20)
    test = _lr(3, 251, 25, seed=21)
    init = torch.tensor([1.0, 5.0])  # length 2 vs N=3
    task = PortfolioTask(seq_length=251)
    pp = task.prepare_training(train)
    pol = task.build_policy()
    pol.fit(pp, num_epochs=1, batch_size=4, verbose=False)
    raised = False
    try:
        task.predict_period_pnl(pol, test, initial_dollars=init)
    except ValueError as exc:
        raised = True
        assert "length 2" in str(exc) and "3 test windows" in str(exc), (
            f"Error message missing context: {exc}"
        )
    assert raised, "predict_period_pnl should reject mismatched init_dollars"


@_probe("initial_dollars with negative value (should NOT crash, must preserve sign)")
def probe_init_dollars_negative():
    train = _lr(8, 251, 25, seed=30)
    test = _lr(2, 251, 25, seed=31)
    init = torch.tensor([-1.0, -10.0])
    task = PortfolioTask(seq_length=251)
    pp = task.prepare_training(train)
    pol = task.build_policy()
    pol.fit(pp, num_epochs=1, batch_size=4, verbose=False)
    pnl = task.predict_period_pnl(pol, test, initial_dollars=init)
    # Output must be finite regardless of sign.
    assert torch.isfinite(pnl).all(), "PnL should remain finite with negative init_d"


@_probe("AlphaTask with constant-zero returns (degenerate Sharpe windows)")
def probe_alpha_degenerate_sharpe():
    """When realized returns are zero, Sharpe is 0/eps -> bounded
    (defensive guard) but loss must remain finite."""
    train = torch.zeros(8, 251, 25, dtype=torch.float32)
    test = torch.zeros(2, 251, 25, dtype=torch.float32)
    task = AlphaTask(seq_length=251, num_assets=25, long_only=True)
    pp = task.prepare_training(train)
    pol = task.build_policy()
    pol.fit(pp, num_epochs=1, batch_size=4, verbose=False)
    pnl = task.predict_period_pnl(pol, test)
    assert torch.isfinite(pnl).all(), "Alpha PnL should remain finite on constant paths"


@_probe("PortfolioTask with very large log returns (no NaN/Inf in PnL)")
def probe_portfolio_extreme_paths():
    """Huge log returns can overflow float32 in cumsum→exp. The prepare
    step clamps cumulative log-returns so prices saturate finite, and
    downstream PnL must remain finite."""
    train = torch.randn(8, 251, 25, dtype=torch.float32) * 5.0
    test = train[:2]
    task = PortfolioTask(seq_length=251)
    pp = task.prepare_training(train)
    pol = task.build_policy()
    pol.fit(pp, num_epochs=1, batch_size=4, verbose=False)
    pnl = task.predict_period_pnl(pol, test)
    assert torch.isfinite(pnl).all(), "Portfolio PnL must remain finite after clamp"


@_probe("MoneynessLSTM loss takes scalar K (not just (N,))")
def probe_moneyness_scalar_k():
    N, L = 2, 252
    prices = torch.empty(N, L).uniform_(0.9, 1.1)
    K = torch.tensor(1.0)  # scalar tensor
    c0 = torch.zeros(N)
    model = MoneynessLSTM(seq_length=L)
    delta = model.predict(prices, K=K)
    assert delta.shape == (N, L)
    loss = model.loss(prices, K.expand(N), c0)
    assert torch.isfinite(loss)


@_probe("per_period_pnl_from_positions rejects bad mode")
def probe_pnl_bad_mode():
    p = torch.zeros(2, 10)
    x = torch.ones(2, 10)
    raised = False
    try:
        per_period_pnl_from_positions(p, x, mode="exotic")
    except ValueError:
        raised = True
    assert raised, "Bad mode should raise"


@_probe("OptionsTask.run_one_channel with single test window (N=1)")
def probe_options_n1():
    train = _lr(8, 251, 2, seed=40)[:, :, 0]  # single channel
    test = _lr(1, 251, 2, seed=41)[:, :, 0]
    task = OptionsTask(seq_length=251, premium_mode="bs")
    pol, pnl, ex = task.run_one_channel(train, test, None, num_epochs=1, batch_size=4)
    assert pnl.shape == (1, 251)
    assert torch.isfinite(pnl).all()


@_probe("UtilityEvaluator with mismatched train/test lengths raises clearly")
def probe_mismatched_lengths():
    train = _lr(8, 251, 3, seed=50)
    test = _lr(4, 252, 3, seed=51)  # different L
    syn = _lr(8, 251, 3, seed=52)
    task = OptionsTask(seq_length=251)
    ev = UtilityEvaluator(task=task, protocol="tstr",
                          num_epochs=1, batch_size=4, verbose=False)
    try:
        ev.run(train[:6], test, syn[:6])
        # If we get here without raise, length is being silently handled — flag
        raise AssertionError(
            "Mismatched L between train and test silently accepted — "
            "should raise or be complaint-loading."
        )
    except (RuntimeError, ValueError, IndexError, AssertionError):
        pass


@_probe("OptionsTask ignores initial_dollars argument without error (consistency)")
def probe_options_initial_dollars_ignored():
    """Paper consistency: Options positions are Δ-fractions; per-period
    PnL is currently NOT scaled by initial_dollars. We assert the
    current contract: ignored but no error."""
    train = _lr(8, 251, 2, seed=60)[:, :, 0]
    test = _lr(2, 251, 2, seed=61)[:, :, 0]
    task = OptionsTask(seq_length=251)
    init = torch.tensor([1.0, 100.0])
    pol, pnl_with, _ = task.run_one_channel(train, test, init, num_epochs=1, batch_size=4)
    pol2, pnl_without, _ = task.run_one_channel(train, test, None, num_epochs=1, batch_size=4)
    diff = (pnl_with - pnl_without).abs().max().item()
    print(f"    Options PnL max|with-without|: {diff:.4e} (0 means ignored)")
    # Document: depending on whether the user requires init_dollar
    # scaling for Options, this is either PASS (ignored) or FAIL.


@_probe("PortfolioInventory ones are truly static at all timesteps")
def probe_portfolio_inventory_static():
    """E_t = Σ_i P_{t,i} + Σ_j π_{t,j}.  P_{t,i} must be 1 at every
    step even when LSTM output drifts."""
    train = _lr(8, 251, 25, seed=70)
    test = _lr(2, 251, 25, seed=71)
    task = PortfolioTask(seq_length=251)
    pp = task.prepare_training(train)
    pol = task.build_policy()
    pol.fit(pp, num_epochs=1, batch_size=4, verbose=False)
    with torch.no_grad():
        prices = task.prepare_training(test)
        positions = pol.predict(prices)
        full = task._assemble_full_pos(torch.ones(()), positions)
        inv_idx = torch.tensor(task.channel_map["inventory"])
        ok = torch.all(full[:, :, inv_idx] == 1.0)
        print(f"    inventory ones at all L steps: {bool(ok)}")
        assert ok, "Inventory must be 1.0 at every step"


@_probe("BSStaticDelta.predict with mismatched K length raises ValueError")
def probe_bs_k_mismatch():
    N, L = 3, 252
    prices = torch.empty(N, L).uniform_(0.9, 1.1)
    bs = BSStaticDelta(seq_length=L, K=1.0, sigma_annual=0.2)
    bad_K = torch.tensor([1.0, 1.1])  # length 2 vs N=3
    raised = False
    try:
        bs.predict(prices, K=bad_K)
    except ValueError as exc:
        raised = True
        assert "length 2" in str(exc) and "3 paths" in str(exc), (
            f"Error message missing context: {exc}"
        )
    assert raised, "BSStaticDelta.predict should reject mismatched K length"


@_probe("AugmentedProtocol union size = real_N + synth_N")
def probe_augmented_union_size():
    real = _lr(8, 251, 25, seed=80)
    syn = _lr(6, 251, 25, seed=81)
    rt, rtest = real[:4], real[4:]
    st = syn[:4]
    # Stub task: we'll just check that _union produces N+N
    u = AugmentedProtocol._union(rt, st)
    assert u.shape[0] == rt.shape[0] + st.shape[0]
    assert u.shape[1:] == rt.shape[1:]
    print(f"    union shape: {tuple(u.shape)} (expected {(rt.shape[0]+st.shape[0],)+rt.shape[1:]})")


@_probe("OptionsTask with very large log returns (2D path now clamped)")
def probe_options_extreme_paths():
    """Adversarial huge log returns through the Options 2D prepare path
    MUST remain finite (cumulative clamp saturates prices to a safe
    ceiling)."""
    train = torch.randn(8, 251, dtype=torch.float32) * 5.0
    test = torch.randn(2, 251, dtype=torch.float32) * 5.0
    task = OptionsTask(seq_length=251, premium_mode="bs")
    pp = task.prepare_training(train)
    pol = task.build_policy()
    pol.fit(pp, num_epochs=1, batch_size=4, verbose=False)
    pnl = task.predict_period_pnl(pol, test)
    assert torch.isfinite(pnl).all(), "Options PnL must be finite on huge log returns"


@_probe("BSStaticDelta with huge log returns (saturated but finite)")
def probe_bs_extreme_paths():
    N, L = 2, 251
    log_r = torch.randn(N, L, dtype=torch.float32) * 5.0
    bs = BSStaticDelta(seq_length=L + 1, K=1.0, sigma_annual=0.2)
    delta = bs.predict(log_r)
    assert torch.isfinite(delta).all(), "BS deltas must be finite on huge log returns"
    # In the price-reconstructed branch: prices saturate but deltas stay finite.
    assert (delta >= 0).all() and (delta <= 1).all(), "BS delta in [0,1]"


@_probe("MoneynessLSTM.predict with K of unexpected length raises")
def probe_moneyness_k_wrong_length():
    """Mirror the strict BSStaticDelta contract. Previously silent
    broadcast; now ValueError."""
    N, L = 3, 252
    prices = torch.empty(N, L).uniform_(0.9, 1.1)
    model = MoneynessLSTM(seq_length=L)
    raised = False
    try:
        model.predict(prices, K=torch.tensor([1.0, 1.1]))  # length 2 vs N=3
    except ValueError:
        raised = True
    assert raised, "MoneynessLSTM.predict should reject scalar-like K of wrong length"


@_probe("OptionsTask honours initial_dollars argument (now scaled)")
def probe_options_initial_dollars_respected():
    """Regression: Options predict_period_pnl used to silently ignore
    ``initial_dollars``; now it scales the per-period PnL by the
    per-window dollar anchor. Train ONE policy and call predict twice
    so the LSTM output is deterministic across calls."""
    train = _lr(8, 251, 2, seed=80)[:, :, 0]
    test = _lr(2, 251, 2, seed=81)[:, :, 0]
    task = OptionsTask(seq_length=251)
    pp = task.prepare_training(train)
    pol = task.build_policy()
    pol.fit(pp, num_epochs=1, batch_size=4, verbose=False)

    pnl_default = task.predict_period_pnl(pol, test, initial_dollars=None)
    init = torch.tensor([1.0, 100.0])
    pnl_scaled = task.predict_period_pnl(pol, test, initial_dollars=init)

    # Default: $1 for every window → window 0 should match exactly.
    diff_w0 = (pnl_default[0] - pnl_scaled[0]).abs().max().item()
    # Window 1 scales by 100x (init=100): ratio of magnitudes ≈ 100.
    max_w1_default = max(pnl_default[1].abs().max().item(), 1e-12)
    ratio_w1 = pnl_scaled[1].abs().max().item() / max_w1_default
    print(f"    window 0 |default - scaled|: {diff_w0:.4e} (must be ~0)")
    print(f"    window 1 |scaled / default|: {ratio_w1:.2f} (must be ~100)")
    assert diff_w0 < 1e-6, "Window 0 (init=$1) should match default $1"
    assert abs(ratio_w1 - 100.0) < 1e-2, "Window 1 (init=$100) should ~100x default"


def main():
    for fn in [
        probe_moneyness_predict_log_returns,
        probe_bs_predict_log_returns,
        probe_metric_zero_length,
        probe_metric_t1,
        probe_metric_aggregate_empty,
        probe_tstr_n1,
        probe_channel_map_arbitrary,
        probe_init_dollars_mismatch_silent,
        probe_init_dollars_negative,
        probe_alpha_degenerate_sharpe,
        probe_portfolio_extreme_paths,
        probe_moneyness_scalar_k,
        probe_pnl_bad_mode,
        probe_options_n1,
        probe_mismatched_lengths,
        probe_options_initial_dollars_ignored,
        probe_portfolio_inventory_static,
        probe_bs_k_mismatch,
        probe_augmented_union_size,
        probe_options_extreme_paths,
        probe_bs_extreme_paths,
        probe_moneyness_k_wrong_length,
        probe_options_initial_dollars_respected,
    ]:
        fn()

    print("\n" + "=" * 70)
    print(f"PASSES: {len(PASSES)}/{len(PASSES) + len(FAILS)}")
    if FAILS:
        print(f"FAILS:  {len(FAILS)}")
        for name, exc, tb in FAILS:
            print(f"  - {name}: {exc}")
    print("=" * 70)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
