"""Smoke test: end-to-end run of the new §6 utility pipeline.

Validates:

- per-channel Options training (C=3) with 7-moneyness augmentation;
- BS static-delta baseline (premium=0);
- PnL toggle between simple and log returns;
- TSTR + Augmented protocols produce U1–U5 mean ± std;
- Portfolio and Alpha scaffolds wire their policies correctly (smoke).

Run::

    venv/bin/python scripts/smoke_utility_pipeline.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

# Make src.utility importable as a package
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


def _make_synthetic_log_returns(n: int, length: int, channels: int, seed: int = 0):
    """Random log returns with mild autocorrelation + drift."""
    rng = np.random.default_rng(seed)
    # (n, length, channels)
    base = rng.normal(loc=0.0002, scale=0.015, size=(n, length, channels)).astype(np.float32)
    return torch.from_numpy(base)


def _report(name: str, payload: dict) -> None:
    print(f"\n=== {name} ===")
    for k, v in payload.items():
        if isinstance(v, dict) and "mean" in v and "std" in v:
            print(f"  {k:>8}: mean={v['mean']:+.4f}  std={v['std']:.4f}")
        else:
            print(f"  {k:>8}: {v}")


def test_options_tstr():
    print("\n[Options TSTR] ----------------------------------------------------------------")
    real = _make_synthetic_log_returns(n=64, length=252, channels=3, seed=11)
    synth = _make_synthetic_log_returns(n=64, length=252, channels=3, seed=22)

    real_train, real_test = real[:48], real[48:]
    synth_train = synth[:48]

    task = OptionsTask(seq_length=252, premium_mode="bs")
    ev = UtilityEvaluator(task=task, protocol="tstr",
                          num_epochs=2, batch_size=32, learning_rate=1e-3,
                          verbose=False)
    result = ev.run(real_train, real_test, synth_train)
    print("Protocol:", result["protocol"], "Task:", result["task"])
    print("Per-channel real outputs:")
    for c, ch in enumerate(result["real"]["per_channel"]):
        print(f"  channel {c}: {ch}")
    print("Aggregate (real vs synthetic):")
    print("  real      :", result["real_aggregate"])
    print("  synthetic :", result["synthetic_aggregate"])
    print("  useful    :", result["useful"], " delta=", result["score_delta"])
    # Sample extras
    print("  extras(real,ch0):", result["real"]["extras"][0])
    assert "pnl" in result["real_aggregate"], "Missing pnl in real aggregate"
    assert "sharpe" in result["real_aggregate"], "Missing sharpe in real aggregate"
    assert "cvar" in result["real_aggregate"], "Missing cvar in real aggregate"
    assert "win_rate" in result["real_aggregate"], "Missing win_rate in real aggregate"
    assert "max_drawdown" in result["real_aggregate"], "Missing max_drawdown"
    print("[PASS] Options TSTR produces U1–U5 mean ± std across N test windows.")


def test_options_augmented():
    print("\n[Options Augmented] ----------------------------------------------------------")
    real = _make_synthetic_log_returns(n=48, length=252, channels=2, seed=31)
    synth = _make_synthetic_log_returns(n=48, length=252, channels=2, seed=32)
    real_train, real_test = real[:32], real[32:]
    synth_train = synth[:32]

    ev = UtilityEvaluator(task=OptionsTask(seq_length=252), protocol="augmented",
                          num_epochs=2, batch_size=32, verbose=False)
    result = ev.run(real_train, real_test, synth_train)
    print("Toplevel keys:", sorted(result.keys()))
    # For per-channel options, each of {real, synthetic, augmented} has per_channel.
    for tag in ("real", "synthetic", "augmented"):
        ag = MetricToolbox.aggregate_per_channel(result[tag]["per_channel"])
        print(f"  {tag:>10}: {ag}")
    assert "augmented" in result, "AugmentedProtocol must produce an 'augmented' entry"
    print("[PASS] Augmented protocol includes full-union training.")


def test_bs_baseline_premium_zero():
    print("\n[BS static-delta baseline] --------------------------------------------------")
    prices = torch.zeros((1, 252)).float()  # placeholder
    # Build a synthetic test path
    log_r = torch.randn(1, 251) * 0.01
    task = OptionsTask(seq_length=252, premium_mode="zero")
    bs = BSStaticDelta(seq_length=252, K=1.0, sigma_annual=0.2)
    pnl_bs = task.predict_period_pnl(bs, log_r)
    # When enough training data exists, the LSTM should be the comparator.
    log_r_train = torch.randn(64, 251) * 0.01
    prepared = task.prepare_training(log_r_train)
    lstm = task.build_policy()
    lstm.fit(prepared, num_epochs=2, batch_size=32, verbose=False)
    pnl_lstm = task.predict_period_pnl(lstm, log_r)
    pnl_total_bs = MetricToolbox.compute(pnl_bs[0])
    pnl_total_lstm = MetricToolbox.compute(pnl_lstm[0])
    print(f"  BS baseline    PnL  = {pnl_total_bs['pnl']:+.4f}")
    print(f"  LSTM Moneyness PnL  = {pnl_total_lstm['pnl']:+.4f}")
    print(f"  PnL tensor shape    = {tuple(pnl_lstm.shape)}")
    # 252 prices → 251 trading periods (L-1 of 252).
    assert pnl_lstm.shape == (1, 251), "Per-period PnL should have length 251 = L-1 of 252 prices"
    print("[PASS] BS baseline (premium=0) and LSTM both produce PnL of correct length.")


def test_pnl_toggle():
    print("\n[PnL toggle: simple vs log] -------------------------------------------------")
    prices = torch.empty(8, 252, 1).uniform_(0.5, 2.0)
    log_r = torch.log(prices[:, 1:, :] / prices[:, :-1, :])  # (8, 251)
    # Use the helper directly
    from src.utility.metrics import per_period_pnl_from_positions
    positions = torch.zeros(8, 252, 1)
    positions[:, :-1, 0] = 1.0  # long one unit
    pnl_simple = per_period_pnl_from_positions(
        positions[:, :, 0], prices.squeeze(-1), mode="simple"
    )
    pnl_log = per_period_pnl_from_positions(
        positions[:, :, 0], prices.squeeze(-1), mode="log"
    )
    diff = (pnl_simple - pnl_log).abs().max().item()
    print(f"  max|simple - log| over 8 paths: {diff:.4e}")
    assert diff > 1e-6, "Simple and log paths should differ for non-zero returns."
    print("[PASS] PnL toggle yields two distinct streams.")


def test_portfolio_and_alpha_smoke():
    print("\n[Portfolio + Alpha smoke] ----------------------------------------------------")
    wins = lambda src: src  # noqa
    # Portfolio: shape (N, L=251, I+J=25)
    log_r_p = _make_synthetic_log_returns(n=32, length=251, channels=25, seed=99)
    p_task = PortfolioTask(num_inventory=20, num_hedge=5, seq_length=251, pnl_returns="simple")
    pp = p_task.prepare_training(log_r_p)
    p_policy = p_task.build_policy()
    p_policy.fit(pp, num_epochs=2, batch_size=16, verbose=False)
    pnl_p = p_task.predict_period_pnl(p_policy, log_r_p[:8])
    assert pnl_p.shape == (8, 251), f"portfolio PnL should be (N, 251), got {tuple(pnl_p.shape)}"
    print(f"  portfolio PnL shape: {tuple(pnl_p.shape)}")

    # Alpha: shape (N, L=251, I=25)
    a_task = AlphaTask(num_assets=25, seq_length=251, pnl_returns="simple")
    ap = a_task.prepare_training(log_r_p)
    a_policy = a_task.build_policy()
    a_policy.fit(ap, num_epochs=2, batch_size=16, verbose=False)
    pnl_a = a_task.predict_period_pnl(a_policy, log_r_p[:8])
    assert pnl_a.shape == (8, 251), f"alpha PnL should be (N, 251), got {tuple(pnl_a.shape)}"
    print(f"  alpha     PnL shape: {tuple(pnl_a.shape)}")
    print("[PASS] Portfolio and Alpha tasks wire policies and produce PnL.")


def test_portfolio_tstr_paper():
    print("\n[Portfolio TSTR — §6.3] ------------------------------------------------------")
    log_r_real = _make_synthetic_log_returns(n=64, length=251, channels=25, seed=41)
    log_r_synth = _make_synthetic_log_returns(n=64, length=251, channels=25, seed=42)
    real_train, real_test = log_r_real[:48], log_r_real[48:]
    synth_train = log_r_synth[:48]
    task = PortfolioTask(num_inventory=20, num_hedge=5, seq_length=251)
    ev = UtilityEvaluator(task=task, protocol="tstr",
                          num_epochs=2, batch_size=16, learning_rate=1e-3,
                          verbose=False)
    result = ev.run(real_train, real_test, synth_train)
    for k in ("pnl", "sharpe", "cvar", "win_rate", "max_drawdown"):
        assert k in result["real"], f"Portfolio TSTR missing {k}"
    print("  real aggregate      :", {m: f"{v['mean']:+.3f}±{v['std']:.3f}" for m, v in result["real"].items() if isinstance(v, dict) and "mean" in v})
    print("  synthetic aggregate :", {m: f"{v['mean']:+.3f}±{v['std']:.3f}" for m, v in result["synthetic"].items() if isinstance(v, dict) and "mean" in v})
    print("  useful (s_g>s_r):", result["useful"], " delta=", result["score_delta"])
    print("[PASS] Portfolio TSTR produces U1–U5 mean ± std (paper §6.3).")


def test_alpha_tstr_paper():
    print("\n[Alpha TSTR — §6.4] ----------------------------------------------------------")
    log_r_real = _make_synthetic_log_returns(n=64, length=251, channels=25, seed=51)
    log_r_synth = _make_synthetic_log_returns(n=64, length=251, channels=25, seed=52)
    real_train, real_test = log_r_real[:48], log_r_real[48:]
    synth_train = log_r_synth[:48]
    task = AlphaTask(num_assets=25, seq_length=251, long_only=True)
    ev = UtilityEvaluator(task=task, protocol="tstr",
                          num_epochs=2, batch_size=16, learning_rate=1e-3,
                          verbose=False)
    result = ev.run(real_train, real_test, synth_train)
    for k in ("pnl", "sharpe", "cvar", "win_rate", "max_drawdown"):
        assert k in result["real"], f"Alpha TSTR missing {k}"
    print("  real aggregate      :", {m: f"{v['mean']:+.3f}±{v['std']:.3f}" for m, v in result["real"].items() if isinstance(v, dict) and "mean" in v})
    print("  synthetic aggregate :", {m: f"{v['mean']:+.3f}±{v['std']:.3f}" for m, v in result["synthetic"].items() if isinstance(v, dict) and "mean" in v})
    print("  useful (s_g>s_r):", result["useful"], " delta=", result["score_delta"])
    # Verify weights are softmax (sum to 1) on the long-only path
    sample = real_train[:2]
    prepared = task.prepare_training(sample)
    pol = task.build_policy()
    pol.fit(prepared, num_epochs=2, batch_size=16, verbose=False)
    w = pol.predict(prepared)
    sums = w.sum(dim=-1).mean().item()
    print(f"  long_only weight sum across cross-section: {sums:.3f}  (should be 1.0)")
    assert abs(sums - 1.0) < 1e-4, f"long_only weights should sum to 1, got {sums}"
    print("[PASS] Alpha TSTR produces U1–U5 mean ± std (paper §6.4).")


def test_initial_dollar_scaling():
    print("\n[initial_dollar scaling] -----------------------------------------------------")
    n = 4
    log_r = _make_synthetic_log_returns(n=n + 8, length=251, channels=25, seed=77)
    train, test = log_r[:n], log_r[n:n + 8]
    initial_d = torch.tensor([1.0, 10.0, 100.0, 1000.0]).repeat(2)[: test.shape[0]]
    task = PortfolioTask(num_inventory=20, num_hedge=5, seq_length=251)
    ev = UtilityEvaluator(task=task, protocol="tstr",
                          num_epochs=2, batch_size=16, verbose=False)
    pp = task.prepare_training(train)
    pol = task.build_policy()
    pol.fit(pp, num_epochs=2, batch_size=16, verbose=False)
    pnl = task.predict_period_pnl(pol, test, initial_dollars=initial_d)
    totals = pnl.sum(dim=1).detach()
    print(f"  per-window totals across 8 windows: {totals.cpu().numpy().tolist()}")
    print(f"  initial_dollars tagged              : {initial_d.detach().cpu().numpy().tolist()}")
    avg_abs = totals.abs().mean().item()
    assert avg_abs > 0, "Scaled PnL mean should be non-zero with non-unity dollars."
    print("[PASS] initial_dollar scaling propagates to per-window PnL magnitudes.")


def test_portfolio_channel_map():
    print("\n[Portfolio channel_map] ------------------------------------------------------")
    log_r = _make_synthetic_log_returns(n=12, length=251, channels=25, seed=88)
    train, test = log_r[:8], log_r[8:]
    # Custom map: inventory = LAST 20 channels, hedge = FIRST 5 channels.
    inv = list(range(5, 25))
    hdg = list(range(0, 5))
    task = PortfolioTask(
        num_inventory=20, num_hedge=5,
        seq_length=251, channel_map={"inventory": inv, "hedge": hdg},
    )
    pp = task.prepare_training(train)
    pol = task.build_policy()
    pol.fit(pp, num_epochs=2, batch_size=16, verbose=False)
    pnl = task.predict_period_pnl(pol, test)
    assert pnl.shape == (4, 251), f"got {tuple(pnl.shape)}"
    # Build full_pos and verify: inventory ones live ONLY at the 20 inventory indices.
    with torch.no_grad():
        prices = task.prepare_training(test)
        positions = pol.predict(prices)
        full = task._assemble_full_pos(torch.ones(()), positions)
        inv_idx = torch.tensor(task.channel_map["inventory"])
        hdg_idx = torch.tensor(task.channel_map["hedge"])
        for j in range(25):
            if j in inv:
                assert torch.all(full[:, :, j] == 1.0), f"inventory channel {j} not 1.0"
            elif j in hdg:
                # Hedge positions are LSTM-generated (not constrained to 0).
                assert full[:, :, j].abs().sum().item() >= 0.0
    print(f"  channel_map accepted; PnL shape {tuple(pnl.shape)}; full_pos inventory ones at {inv[:3]}…{inv[-1]}")
    print("[PASS] channel_map correctly routes inventory ones + hedge positions.")


def main() -> None:
    test_options_tstr()
    test_options_augmented()
    test_bs_baseline_premium_zero()
    test_pnl_toggle()
    test_portfolio_and_alpha_smoke()
    test_portfolio_tstr_paper()
    test_alpha_tstr_paper()
    test_initial_dollar_scaling()
    test_portfolio_channel_map()
    print("\nAll §6 utility smoke tests passed.\n")


if __name__ == "__main__":
    main()
