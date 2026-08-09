"""Build the Portfolio (§6.3) and Alpha (§6.4) smoke-test notebooks.

Produces:
- notebooks/02_portfolio_hedge_smoke.ipynb
- notebooks/03_alpha_gen_smoke.ipynb

Both empty-output (so the user runs them interactively in Jupyter).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Users/uyenlamho/Documents/vscode/CSCD94F25/Unified-benchmark-for-SDGFTS")
NB_DIR = ROOT / "notebooks"


# --------------------------------------------------------------------------- #
# Cell helpers                                                                #
# --------------------------------------------------------------------------- #
def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines],
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


def _kernelspec() -> dict:
    return {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _wrap(cells: list[dict]) -> dict:
    nb = {"cells": cells}
    nb.update(_kernelspec())
    return nb


# --------------------------------------------------------------------------- #
# Shared prelude                                                              #
# --------------------------------------------------------------------------- #
def _imports_cell() -> dict:
    return code(
        "import sys",
        "sys.path.insert(0, '/Users/uyenlamho/Documents/vscode/CSCD94F25/Unified-benchmark-for-SDGFTS')",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import torch",
        "import matplotlib.pyplot as plt",
        "",
        "from src.utility import (",
        "    MetricToolbox, UtilityEvaluator,",
        f")",
        "print('Imports OK; utils loaded from src.utility')",
    )


def _make_log_returns_cell(n_real: int, n_test: int, n_synth: int, length: int,
                            channels: int, real_seed: int, synth_seed: int,
                            real_mu: float, real_sigma: float,
                            synth_mu: float, synth_sigma: float) -> list[dict]:
    return [
        code(
            f"def make_log_returns(n, L, C, mu, sigma, seed):",
            f"    rng = np.random.default_rng(seed)",
            f"    return torch.from_numpy(",
            f"        rng.normal(loc=mu, scale=sigma, size=(n, L, C)).astype(np.float32)",
            f"    )",
            "",
            f"N_TRAIN_REAL, N_TEST, N_TRAIN_SYN = {n_real}, {n_test}, {n_synth}",
            f"SEQ_LEN, N_CHANNELS = {length}, {channels}",
            "",
            f"real_train = make_log_returns(N_TRAIN_REAL, SEQ_LEN, N_CHANNELS, mu={real_mu:.1e}, sigma={real_sigma}, seed={real_seed})",
            f"real_test  = make_log_returns(N_TEST,      SEQ_LEN, N_CHANNELS, mu={real_mu:.1e}, sigma={real_sigma}, seed={real_seed + 1})",
            f"syn_train  = make_log_returns(N_TRAIN_SYN, SEQ_LEN, N_CHANNELS, mu={synth_mu:.1e}, sigma={synth_sigma}, seed={synth_seed})",
            "",
            "for name, t in [('real_train', real_train), ('real_test', real_test), ('syn_train', syn_train)]:",
            "    print(f'{name:<11}: shape={tuple(t.shape)}  mean={t.mean().item():+.4e}  std={t.std().item():.4f}')",
        ),
    ]


# --------------------------------------------------------------------------- #
# Portfolio notebook                                                          #
# --------------------------------------------------------------------------- #
def portfolio_cells() -> list[dict]:
    cells: list[dict] = []
    cells.append(md(
        "# StonkBench §6.3 — Portfolio Hedging Smoke Test",
        "",
        "This notebook runs an end-to-end smoke test of the **§6.3 Portfolio ",
        "Hedging** task from the new utility pipeline on synthetic fake data ",
        "(log-return windows over `I = 20` inventory stocks and `J = 5` ",
        "hedge ETFs).",
        "",
        "It exercises: `PortfolioTask` with one paper-faithful ",
        "`PortfolioLSTM` policy, the **TSTR** protocol (§6.1.1) and ",
        "**Augmented** protocol (§6.1.2), the **U1–U5 PnL toolbox** (§6.1.3),",
        "the `channel_map` interface and the **per-window `initial_dollar`**",
        "scaling.",
        "",
        "Pipeline architecture: scan or open ",
        "`docs/utility_pipeline.png`.",
    ))
    cells.append(md(
        "## How the Portfolio pipeline works",
        "",
        "* **Task** `PortfolioTask` consumes the *whole* multi-asset window ",
        "  ``(N, L, I+J=25)``. The first `I` channels are static unit ",
        "  inventory ``P_{t,i} = 1``; the last `J` channels are free hedge ",
        "  ETF positions ``π_{t,j}``.",
        "* **Policy** `PortfolioLSTM` is a paper-faithful LSTM: receives the ",
        "  full window's prices, emits one hedge position per step. Loss is ",
        "  the cumulative absolute net exposure ``L = Σ_t |E_t|`` with ",
        "  ``E_t = I + Σ_j π_{t,j}`` averaged over the batch.",
        "* **Channel mapping** is plumbed through `_assemble_full_pos`: a ",
        "  user can pass any ``channel_map = {\"inventory\": [...], ",
        "  \"hedge\": [...]}`` and the inventory ones live at those indices.",
        "* **Initial dollar** scaling: each test window is tagged with its ",
        "  own starting $ amount; the PnL is scaled linearly.",
        "",
        "Switching PnL convention (`simple` vs `log`) and reporting U1–U5 ",
        "is identical to the Options path.",
    ))
    cells.append(_imports_cell())
    cells.extend(_make_log_returns_cell(
        n_real=64, n_test=32, n_synth=64, length=252, channels=25,
        real_seed=11, synth_seed=22,
        real_mu=2e-4, real_sigma=0.015, synth_mu=5e-4, synth_sigma=0.020,
    ))
    cells.append(md(
        "### Sample price paths",
        "",
        "Reconstruct prices from log returns on a few test windows across ",
        "the first inventory and first hedge channels to verify the input ",
        "distribution.",
    ))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))",
        "# First inventory channel: index 0",
        "for i in range(5):",
        "    p = torch.exp(torch.cumsum(real_test[:5, :, 0], dim=1))[i].numpy()",
        "    axes[0].plot(p, alpha=0.65, label=f'path {i}')",
        "# First hedge channel: index I (= 20)",
        "I, J = 20, 5",
        "for i in range(5):",
        "    p = torch.exp(torch.cumsum(real_test[:5, :, I], dim=1))[i].numpy()",
        "    axes[1].plot(p, alpha=0.65, label=f'path {i}')",
        "axes[0].set_title('Inventory channel 0 (real test)'); axes[0].set_xlabel('trading day'); axes[0].set_ylabel('price (S_0 = 1)')",
        "axes[1].set_title(f'Hedge channel {I} (real test)'); axes[1].set_xlabel('trading day')",
        "axes[0].legend(fontsize=8, loc='upper left')",
        "plt.tight_layout(); plt.show()",
    ))
    cells.append(md(
        "## 1. Portfolio task — TSTR protocol (§6.1.1)",
        "",
        "TSTR trains `M_r` on real and `M̂_g` on synthetic, scores both on ",
        "`D_test`. Useful iff `s_g > s_r`.",
    ))
    cells.append(code(
        "from src.utility import PortfolioTask",
        "",
        "task = PortfolioTask(num_inventory=20, num_hedge=5, seq_length=252, pnl_returns='simple')",
        "ev_tstr = UtilityEvaluator(task=task, protocol='tstr',",
        "                           num_epochs=8, batch_size=16, learning_rate=1e-3, verbose=False)",
        "tstr_result = ev_tstr.run(real_train, real_test, syn_train)",
        "print('Protocol:', tstr_result['protocol'], '| Task:', tstr_result['task'])",
        "print('Useful (s_g > s_r):', tstr_result['useful'], '| delta =', f\"{tstr_result['score_delta']:+.4f}\")",
    ))
    cells.append(md(
        "### Paper-style table (Table 1)",
        "",
        "Real vs synthetic aggregate U1–U5 (mean ± std across the 32 test ",
        "windows). A *higher U1 PnL* is better per §6.1.1.",
    ))
    cells.append(code(
        "def fmt(agg, key):",
        "    return f\"{agg[key]['mean']:+.3f} ± {agg[key]['std']:.3f}\"",
        "",
        "real_agg = tstr_result['real']",
        "syn_agg  = tstr_result['synthetic']",
        "rows = []",
        "for source, agg in [('Real (M_r)', real_agg), ('Synthetic (M̂_g)', syn_agg)]:",
        "    rows.append({",
        "        'training source': source,",
        "        'U1 PnL': fmt(agg, 'pnl'),",
        "        'U2 Sharpe': fmt(agg, 'sharpe'),",
        "        'U3 CVaR α=0.05': fmt(agg, 'cvar'),",
        "        'U4 Win rate': fmt(agg, 'win_rate'),",
        "        'U5 Max DD': fmt(agg, 'max_drawdown'),",
        "    })",
        "df1 = pd.DataFrame(rows).set_index('training source')",
        "df1.style.set_caption(",
        "    \"Table 1 — Portfolio task (§6.3) U1–U5, real vs synthetic (32 test windows, mean ± std)\"",
        ").format(precision=3)",
    ))
    cells.append(md(
        "## 2. Augmented training protocol (§6.1.2)",
        "",
        "Augmented full-union training `M̃` on `D_train ∪ D̂_g` (no ",
        "balancing). All three are scored on `D_test`.",
    ))
    cells.append(code(
        "ev_aug = UtilityEvaluator(task=task, protocol='augmented',",
        "                          num_epochs=8, batch_size=16, learning_rate=1e-3, verbose=False)",
        "aug_result = ev_aug.run(real_train, real_test, syn_train)",
        "",
        "def agg(payload):",
        "    \"\"\"Drill through ``AugmentedProtocol.run()`` shapes: per-channel tasks give",
        "    ``{per_channel, extras}`` (drill into ``per_channel`` and aggregate over channels);",
        "    whole-window tasks give ``{aggregate, extras}`` (use ``aggregate`` directly).",
        "    Without this drilling, ``MetricToolbox.aggregate`` would see an empty filter and",
        "    emit all-NaN values.\"\"\"",
        "    if 'per_channel' in payload:",
        "        return MetricToolbox.aggregate_per_channel(payload['per_channel'])",
        "    if 'aggregate' in payload:",
        "        return payload['aggregate']",
        "    return MetricToolbox.aggregate([payload])",
        "",
        "rows = []",
        "for label, key in [('Real (M_r)', 'real'),",
        "                   ('Synthetic (M̂_g)', 'synthetic'),",
        "                   ('Augmented (M̃, full union)', 'augmented')]:",
        "    ag = agg(aug_result[key])",
        "    rows.append({",
        "        'training source': label,",
        "        'U1 PnL': fmt(ag, 'pnl'),",
        "        'U2 Sharpe': fmt(ag, 'sharpe'),",
        "        'U3 CVaR α=0.05': fmt(ag, 'cvar'),",
        "        'U4 Win rate': fmt(ag, 'win_rate'),",
        "        'U5 Max DD': fmt(ag, 'max_drawdown'),",
        "    })",
        "df2 = pd.DataFrame(rows).set_index('training source')",
        "df2.style.set_caption(",
        "    \"Table 2 — Portfolio task (§6.3) U1–U5, Augmented protocol (full-union D_train ∪ D̂_g)\"",
        ").format(precision=3)",
    ))
    cells.append(md(
        "## 3. `channel_map` inversion regression",
        "",
        "Default ordering puts inventory first (channels `0..19`) and hedge ",
        "last (`20..24`). Here we **flip** the ordering — inventory at ",
        "`5..24`, hedge at `0..4` — and re-train to confirm the task really ",
        "honours the user-supplied map.",
    ))
    cells.append(code(
        "inv = list(range(5, 25))",
        "hdg = list(range(0, 5))",
        "flipped_task = PortfolioTask(",
        "    num_inventory=20, num_hedge=5, seq_length=252,",
        "    channel_map={'inventory': inv, 'hedge': hdg},",
        ")",
        "print('flipped channel_map:', flipped_task.channel_map)",
        "",
        "pp = flipped_task.prepare_training(real_train[:8])",
        "pol = flipped_task.build_policy()",
        "pol.fit(pp, num_epochs=2, batch_size=16, verbose=False)",
        "with torch.no_grad():",
        "    prices = flipped_task.prepare_training(real_test[:4])",
        "    positions = pol.predict(prices)",
        "    full = flipped_task._assemble_full_pos(torch.ones(()), positions)",
        "    # Inventory ones at user-specified indices only",
        "    for j in range(25):",
        "        if j in inv:",
        "            assert torch.all(full[:, :, j] == 1.0), f'inventory channel {j} not 1.0'",
        "        elif j in hdg:",
        "            # hedge positions are LSTM-generated",
        "            pass",
        "print('flipped full_pos inventory columns = 1.0 ✓')",
    ))
    cells.append(md(
        "## 4. `initial_dollar` per-window scaling",
        "",
        "Each test window is tagged with its own starting $ amount.",
        "Per-window PnL is scaled linearly so the dollar magnitudes line up ",
        "with the tagged `initial_dollar` per window.",
    ))
    cells.append(code(
        "from torch import tensor",
        "",
        "n_test = real_test.shape[0]",
        "initial_d = tensor([1.0, 10.0, 100.0, 1000.0]).repeat((n_test // 4) + 1)[:n_test]",
        "",
        "policy = task.build_policy()",
        "policy.fit(task.prepare_training(real_train), num_epochs=2, batch_size=16, verbose=False)",
        "pnl = task.predict_period_pnl(policy, real_test, initial_dollars=initial_d)",
        "totals = pnl.sum(dim=1).detach()",
        "print('per-window totals:    ', totals.cpu().numpy().round(2).tolist())",
        "print('initial_dollars tag:  ', initial_d.cpu().numpy().tolist())",
    ))
    cells.append(md(
        "## 5. Wrap-up",
        "",
        "* **TSTR (§6.1.1)** Woodsum shows the real/synth comparison per ",
        "  generator. The `useful` flag flips covariant with the betterness ",
        "  of `M̂_g` over `M_r`.",
        "* **Augmented (§6.1.2)** adds `M̃` from the full-union training set ",
        "  `D_train ∪ D̂_g`.",
        "* **U1–U5** are aggregated as mean ± std across the N test windows.",
        "* **`channel_map` plumbing** really routes positions through the ",
        "  user-supplied inventory/hedge index sets.",
        "* **`initial_dollar`** scales per-window PnL linearly so dollar-PnL ",
        "  reporting matches the user's tag.",
        "",
        "Next: with real data, this notebook becomes the §6.3 leaderboard ",
        "tab in the StonkBench paper.",
    ))
    return cells


# --------------------------------------------------------------------------- #
# Alpha notebook                                                              #
# --------------------------------------------------------------------------- #
def alpha_cells() -> list[dict]:
    cells: list[dict] = []
    cells.append(md(
        "# StonkBench §6.4 — Alpha Generation Smoke Test",
        "",
        "This notebook runs an end-to-end smoke test of the **§6.4 Alpha ",
        "Generation** task on synthetic fake data (cross-sectional log-return ",
        "windows over `I = 25` assets).",
        "",
        "It exercises: `AlphaTask` with one paper-faithful `AlphaLSTM` ",
        "policy, the **TSTR** protocol (§6.1.1), the **Augmented** protocol ",
        "(§6.1.2), the **U1–U5 PnL toolbox** (§6.1.3), the `long_only` vs ",
        "`long_short` softmax-vs-tanh toggle and per-window `initial_dollar` ",
        "scaling.",
        "",
        "Pipeline architecture: scan or open ",
        "`docs/utility_pipeline.png`.",
    ))
    cells.append(md(
        "## How the Alpha pipeline works",
        "",
        "* **Task** `AlphaTask` consumes the *whole* multi-asset window ",
        "  ``(N, L, I=25)``. There is no inventory/hedge distinction in ",
        "  the asset universe — all `I = 25` channels are tradable.",
        "* **Policy** `AlphaLSTM` is a paper-faithful cross-sectional LSTM: ",
        "  at each step it outputs per-asset weights ``π_t ∈ ℝᴵ`` . Loss is ",
        "  the **negative differentiable Sharpe ratio** ``L = -mean_window(μ/σ)``",
        "  where ``r̂_t = π_{t-1} · r_t`` is the realised per-period portfolio ",
        "  return.",
        "* **`long_only=True`** (paper default) constrains the weights to ",
        "  softmax so they sum to 1. **`long_only=False`** falls back to ",
        "  `tanh` for long-short in `[-1, +1]`.",
        "* **`initial_dollar`** per-window scaling: each test window can be ",
        "  tagged with its own starting $ amount and the per-period PnL is ",
        "  scaled linearly.",
        "",
        "Switching PnL convention (`simple` vs `log`) and reporting U1–U5 ",
        "is identical to the Options/Portfolio path.",
    ))
    cells.append(_imports_cell())
    cells.extend(_make_log_returns_cell(
        n_real=64, n_test=32, n_synth=64, length=252, channels=25,
        real_seed=51, synth_seed=52,
        real_mu=3e-4, real_sigma=0.020, synth_mu=6e-4, synth_sigma=0.025,
    ))
    cells.append(md(
        "### Sample price paths",
        "",
        "Plot the first 5 reconstructed price paths for two of the 25 ",
        "assets to confirm inputs are well-formed.",
    ))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))",
        "for i in range(5):",
        "    p = torch.exp(torch.cumsum(real_test[:5, :, 0], dim=1))[i].numpy()",
        "    axes[0].plot(p, alpha=0.65, label=f'path {i}')",
        "    p = torch.exp(torch.cumsum(real_test[:5, :, 12], dim=1))[i].numpy()",
        "    axes[1].plot(p, alpha=0.65, label=f'path {i}')",
        "axes[0].set_title('Asset 0 (real test)'); axes[0].set_xlabel('trading day'); axes[0].set_ylabel('price (S_0=1)')",
        "axes[1].set_title('Asset 12 (real test)'); axes[1].set_xlabel('trading day')",
        "axes[0].legend(fontsize=8, loc='upper left')",
        "plt.tight_layout(); plt.show()",
    ))
    cells.append(md(
        "## 1. Alpha task — TSTR protocol (§6.1.1)",
        "",
        "TSTR trains `M_r` on real and `M̂_g` on synthetic, scores both on ",
        "`D_test`. Useful iff `s_g > s_r`.",
    ))
    cells.append(code(
        "from src.utility import AlphaTask",
        "",
        "task = AlphaTask(num_assets=25, seq_length=252, long_only=True, pnl_returns='simple')",
        "ev_tstr = UtilityEvaluator(task=task, protocol='tstr',",
        "                           num_epochs=8, batch_size=16, learning_rate=1e-3, verbose=False)",
        "tstr_result = ev_tstr.run(real_train, real_test, syn_train)",
        "print('Protocol:', tstr_result['protocol'], '| Task:', tstr_result['task'])",
        "print('Useful (s_g > s_r):', tstr_result['useful'], '| delta =', f\"{tstr_result['score_delta']:+.4f}\")",
    ))
    cells.append(md(
        "### Paper-style table (Table 1)",
        "",
        "Real vs synthetic aggregate U1–U5 (mean ± std across the 32 test ",
        "windows). Higher U1 PnL is the §6.1.1 win condition.",
    ))
    cells.append(code(
        "def fmt(agg, key):",
        "    return f\"{agg[key]['mean']:+.3f} ± {agg[key]['std']:.3f}\"",
        "",
        "real_agg = tstr_result['real']",
        "syn_agg  = tstr_result['synthetic']",
        "rows = []",
        "for source, agg in [('Real (M_r)', real_agg), ('Synthetic (M̂_g)', syn_agg)]:",
        "    rows.append({",
        "        'training source': source,",
        "        'U1 PnL': fmt(agg, 'pnl'),",
        "        'U2 Sharpe': fmt(agg, 'sharpe'),",
        "        'U3 CVaR α=0.05': fmt(agg, 'cvar'),",
        "        'U4 Win rate': fmt(agg, 'win_rate'),",
        "        'U5 Max DD': fmt(agg, 'max_drawdown'),",
        "    })",
        "df1 = pd.DataFrame(rows).set_index('training source')",
        "df1.style.set_caption(",
        "    \"Table 1 — Alpha task (§6.4) U1–U5, real vs synthetic (32 test windows, mean ± std)\"",
        ").format(precision=3)",
    ))
    cells.append(md(
        "### Sharpe diagnostic",
        "",
        "Prints the realised Sharpe ratio of the trained policy's portfolio ",
        "returns. This is what the paper's \"negative differentiable Sharpe\" ",
        "loss is maximising.",
    ))
    cells.append(code(
        "print('alpha extras (real):', tstr_result['real'] if isinstance(tstr_result['real'], dict) and 'sharpe' in tstr_result['real'] else 'see Table 1')",
        "sharpe_real = tstr_result['real']['sharpe']['mean']",
        "sharpe_syn  = tstr_result['synthetic']['sharpe']['mean']",
        "print(f'  per-window realised Sharpe (real): {sharpe_real:+.4f}')",
        "print(f'  per-window realised Sharpe (synth): {sharpe_syn:+.4f}')",
    ))
    cells.append(md(
        "## 2. Augmented training protocol (§6.1.2)",
        "",
        "Augmented full-union training on `D_train ∪ D̂_g`, scored on ",
        "`D_test`.",
    ))
    cells.append(code(
        "ev_aug = UtilityEvaluator(task=task, protocol='augmented',",
        "                          num_epochs=8, batch_size=16, learning_rate=1e-3, verbose=False)",
        "aug_result = ev_aug.run(real_train, real_test, syn_train)",
        "",
        "def _topagg(payload):",
        "    \"\"\"Drill through AugmentedProtocol shapes: per-channel tasks give",
        "    ``{per_channel, extras}`` (collapse via aggregate_per_channel); whole-window",
        "    tasks give ``{aggregate, extras}`` (use the aggregate dict directly). Without",
        "    this branching the helper iterates the wrong things and emits all-NaN rows.\"\"\"",
        "    if 'per_channel' in payload:",
        "        return MetricToolbox.aggregate_per_channel(payload['per_channel'])",
        "    if 'aggregate' in payload:",
        "        return payload['aggregate']",
        "    return MetricToolbox.aggregate([payload])",
        "",
        "rows = []",
        "for label, key in [('Real (M_r)', 'real'),",
        "                   ('Synthetic (M̂_g)', 'synthetic'),",
        "                   ('Augmented (M̃, full union)', 'augmented')]:",
        "    ag = _topagg(aug_result[key])",
        "    rows.append({",
        "        'training source': label,",
        "        'U1 PnL': fmt(ag, 'pnl'),",
        "        'U2 Sharpe': fmt(ag, 'sharpe'),",
        "        'U3 CVaR α=0.05': fmt(ag, 'cvar'),",
        "        'U4 Win rate': fmt(ag, 'win_rate'),",
        "        'U5 Max DD': fmt(ag, 'max_drawdown'),",
        "    })",
        "df2 = pd.DataFrame(rows).set_index('training source')",
        "df2.style.set_caption(",
        "    \"Table 2 — Alpha task (§6.4) U1–U5, Augmented protocol (full-union D_train ∪ D̂_g)\"",
        ").format(precision=3)",
    ))
    cells.append(md(
        "## 3. Cross-sectional weight sanity check",
        "",
        "For the paper default `long_only=True`, the weights should sum to 1 ",
        "at every time step. We sample a trained policy and verify.",
    ))
    cells.append(code(
        "policy = task.build_policy()",
        "policy.fit(task.prepare_training(real_train[:32]), num_epochs=4, batch_size=16, verbose=False)",
        "w = policy.predict(task.prepare_training(real_test[:4]))",
        "sums_per_step = w.sum(dim=-1).mean().item()",
        "print(f'mean Σ_j w_t over time steps and assets: {sums_per_step:.6f}  (should be 1.0 in long_only mode)')",
        "assert abs(sums_per_step - 1.0) < 1e-4",
    ))
    cells.append(md(
        "## 4. `long_only` vs long-short comparison",
        "",
        "Toggle the cross-sectional constraint: `long_only=False` switches ",
        "from softmax to `tanh`, allowing per-asset weights in `[-1, +1]`.",
    ))
    cells.append(code(
        "rows = []",
        "for tag, cfg in [('long_only (softmax)', {'long_only': True}),",
        "                ('long_short (tanh)', {'long_only': False})]:",
        "    t = AlphaTask(num_assets=25, seq_length=252, **cfg, pnl_returns='simple')",
        "    ev = UtilityEvaluator(task=t, protocol='tstr',",
        "                          num_epochs=4, batch_size=16, learning_rate=1e-3, verbose=False)",
        "    r = ev.run(real_train, real_test, syn_train)",
        "    rows.append({",
        "        'weight mode': tag,",
        "        'U1 PnL': fmt(r['real'], 'pnl'),",
        "        'U2 Sharpe': fmt(r['real'], 'sharpe'),",
        "        'U4 Win rate': fmt(r['real'], 'win_rate'),",
        "    })",
        "df3 = pd.DataFrame(rows).set_index('weight mode')",
        "df3.style.set_caption(",
        "    \"Table 3 — cross-sectional constraint comparison on 'real' training\"",
        ").format(precision=3)",
    ))
    cells.append(md(
        "## 5. `initial_dollar` per-window scaling",
        "",
        "Tag four test windows with $1, $10, $100, $1000 and observe that ",
        "the per-window summed PnL scales linearly.",
    ))
    cells.append(code(
        "from torch import tensor",
        "",
        "n_test = real_test.shape[0]",
        "initial_d = tensor([1.0, 10.0, 100.0, 1000.0]).repeat((n_test // 4) + 1)[:n_test]",
        "",
        "policy = task.build_policy()",
        "policy.fit(task.prepare_training(real_train), num_epochs=2, batch_size=16, verbose=False)",
        "pnl = task.predict_period_pnl(policy, real_test, initial_dollars=initial_d)",
        "totals = pnl.sum(dim=1).detach()",
        "print('per-window totals: ', totals.cpu().numpy().round(2).tolist())",
        "print('initial_dollars tag:', initial_d.cpu().numpy().tolist())",
    ))
    cells.append(md(
        "## 6. Wrap-up",
        "",
        "* **TSTR (§6.1.1)** compares `M_r` vs `M̂_g` on `D_test`. The `useful` ",
        "  flag is `s_g > s_r` on U1 PnL by default.",
        "* **Augmented (§6.1.2)** adds the full-union-trained `M̃`.",
        "* **U1–U5** are aggregated as mean ± std across the 32 test windows ",
        "  exactly per paper eq. 27.",
        "* **Cross-sectional softmax** ensures the long-only weights sum to 1 ",
        "  each step (paper default).",
        "* **`initial_dollar`** per-window scaling works identically across ",
        "  all three tasks.",
        "",
        "Next: with real data, this notebook is the §6.4 leaderboard tab.",
    ))
    return cells


# --------------------------------------------------------------------------- #
# Notebook assembly                                                            #
# --------------------------------------------------------------------------- #
def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ("02_portfolio_hedge_smoke.ipynb", portfolio_cells()),
        ("03_alpha_gen_smoke.ipynb", alpha_cells()),
    ]
    for name, cells in targets:
        nb = _wrap(cells)
        path = NB_DIR / name
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        print(f"Wrote {path}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
