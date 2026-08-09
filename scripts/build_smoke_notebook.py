"""Build the §6 utility-pipeline smoke-test notebook.

Constructs ``notebooks/01_utility_pipeline_smoke.ipynb`` as raw JSON. We use
the .ipynb v4 schema directly because ``nbformat`` is not installed in
this venv; ``jupyter nbconvert --execute`` is then used to populate the
cell outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Users/uyenlamho/Documents/vscode/CSCD94F25/Unified-benchmark-for-SDGFTS")
OUTPUT = ROOT / "notebooks" / "01_utility_pipeline_smoke.ipynb"


# --------------------------------------------------------------------------- #
# Cell builders                                                                #
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


# --------------------------------------------------------------------------- #
# Cells                                                                         #
# --------------------------------------------------------------------------- #
def build_cells() -> list[dict]:
    cells: list[dict] = []

    # ---- Title ----------------------------------------------------------- #
    cells.append(md(
        "# StonkBench §6 — Utility Pipeline Smoke Test",
        "",
        "This notebook runs an end-to-end smoke test of the new modular ",
        "**§6 utility pipeline** on synthetic fake data (log-return windows). ",
        "It exercises:",
        "",
        "* the **Options** task (§6.2) with 7-moneyness augmentation and ",
        "  Black–Scholes–Merton premium,",
        "* the **TSTR** protocol (§6.1.1) and **Augmented** protocol (§6.1.2),",
        "* the **U1–U5 PnL toolbox** (§6.1.3) aggregated as mean ± std over ",
        "  the N held-out test windows.",
        "",
        "Pipeline architecture: scan or open ",
        "`docs/utility_pipeline.png` for the full picture.",
    ))

    cells.append(md(
        "## How the pipeline works",
        "",
        "The pipeline is **task × policy × protocol**:",
        "",
        "* **Task** — prepares data, defines per-period PnL:",
        "  * `OptionsTask`: per-channel processing of an `(N, L, C)` tensor, ",
        "    7-moneyness augmentation (§6.2.2).",
        "  * `PortfolioTask`: whole `(L, I+J)` window with I=20 + J=5 (§6.3).",
        "  * `AlphaTask`: whole `(L, I=25)` cross-sectional weights (§6.4).",
        "* **Policy** — one paper-faithful model per task plus a ",
        "  `BSStaticDelta` reference baseline (premium = 0):",
        "  * `MoneynessLSTM`: autoregressive, input `(Δ{t-1}, g̃(t), τ(t))` (§6.2.3).",
        "  * `PortfolioLSTM`: multivariate with loss = Σ|net exposure|.",
        "  * `AlphaLSTM`: cross-sectional softmax weights, loss = −diff Sharpe.",
        "* **Protocol** —-",
        "  * `TSTRProtocol` (§6.1.1): train `M_r` on real, `M̂_g` on synthetic, ",
        "    score both on `D_test`. Useful iff `s_g > s_r`.",
        "  * `AugmentedProtocol` (§6.1.2): same plus `M̃` on the full union ",
        "    `D_train ∪ D̂_g` (no balancing), also scored on `D_test`.",
        "",
        "Each protocol emits per-window U1–U5 metrics; the toolbox then ",
        "aggregates them as **mean ± std across the N test windows** (eq. 27).",
        "",
        "Switching between **simple** and **log** returns for the per-period PnL ",
        "is a single flag (`pnl_returns` on each task).",
    ))

    # ---- Imports --------------------------------------------------------- #
    cells.append(code(
        "import sys, os",
        "sys.path.insert(0, '/Users/uyenlamho/Documents/vscode/CSCD94F25/Unified-benchmark-for-SDGFTS')",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import torch",
        "import matplotlib.pyplot as plt",
        "",
        "from src.utility import (",
        "    OptionsTask, PortfolioTask, AlphaTask,",
        "    MoneynessLSTM, BSStaticDelta, PortfolioLSTM, AlphaLSTM,",
        "    MetricToolbox, TSTRProtocol, AugmentedProtocol, UtilityEvaluator,",
        ")",
        "from src.utility.metrics import per_period_pnl_from_positions",
        "",
        "print('Imports OK; utils loaded from src.utility')",
    ))

    # ---- Synthetic data --------------------------------------------------- #
    cells.append(md(
        "## 1. Generate fake log-return windows",
        "",
        "We construct two synthetic universes to stand in for `D_train` ",
        "(real) and `D̂_g` (synthetic-generator output).",
        "",
        "* **Real** — log returns drawn from μ=2 bps, σ=1.5%.",
        "* **Synthetic** — drawn from μ=5 bps, σ=2.0%.",
        "",
        "We use **3 channels** so per-channel training is visible.",
    ))

    cells.append(code(
        "def make_log_returns(n, L, C, mu, sigma, seed):",
        "    rng = np.random.default_rng(seed)",
        "    return torch.from_numpy(",
        "        rng.normal(loc=mu, scale=sigma, size=(n, L, C)).astype(np.float32)",
        "    )",
        "",
        "N_TRAIN_REAL, N_TEST, N_TRAIN_SYN = 64, 32, 64",
        "SEQ_LEN, N_CHANNELS = 252, 3",
        "",
        "real_train = make_log_returns(N_TRAIN_REAL, SEQ_LEN - 1, N_CHANNELS, mu=2e-4, sigma=0.015, seed=11)",
        "real_test  = make_log_returns(N_TEST,      SEQ_LEN - 1, N_CHANNELS, mu=2e-4, sigma=0.015, seed=12)",
        "syn_train  = make_log_returns(N_TRAIN_SYN, SEQ_LEN - 1, N_CHANNELS, mu=5e-4, sigma=0.020, seed=22)",
        "",
        "for name, t in [('real_train', real_train), ('real_test', real_test), ('syn_train', syn_train)]:",
        "    print(f'{name:<11}: shape={tuple(t.shape)}  mean={t.mean().item():+.4e}  std={t.std().item():.4f}')",
    ))

    # ---- Path preview ---------------------------------------------------- #
    cells.append(md(
        "### Sample paths",
        "",
        "Plot the first 5 reconstructed price paths on channels 0 and 1 of the ",
        "`D_test` to confirm the inputs are sane.",
    ))

    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))",
        "for i in range(5):",
        "    axes[0].plot(torch.exp(torch.cumsum(real_test[:5, :, 0], dim=1))[i].numpy(), alpha=0.65, label=f'path {i}')",
        "    axes[1].plot(torch.exp(torch.cumsum(real_test[:5, :, 1], dim=1))[i].numpy(), alpha=0.65, label=f'path {i}')",
        "axes[0].set_title('Real test, channel 0'); axes[0].set_xlabel('trading day'); axes[0].set_ylabel('price (S_0 = 1)')",
        "axes[1].set_title('Real test, channel 1'); axes[1].set_xlabel('trading day')",
        "axes[0].legend(fontsize=8, loc='upper left')",
        "plt.tight_layout(); plt.show()",
    ))

    # ---- Section: Options TSTR ------------------------------------------- #
    cells.append(md(
        "## 2. Options task — TSTR protocol (§6.1.1)",
        "",
        "The Options task **trains one model per channel** with ",
        "**7-moneyness augmentation**: every input path is replicated at ",
        "`7` strikes `K_j = S_0 / g̃(0)_j` for `g̃(0) ∈ {0.7, 0.8, 0.9, 1.0, ",
        "1.1, 1.2, 1.3}` so the network sees the whole moneyness range. Each ",
        "sub-example gets its own Black–Scholes–Merton premium `c_0(K_j, σ, T)`.",
        "",
        "TSTR runs `M_r` on real and `M̂_g` on synthetic, scores both on ",
        "`D_test`, and reports U1–U5.",
    ))

    cells.append(code(
        "options_task = OptionsTask(seq_length=SEQ_LEN, premium_mode='bs', pnl_returns='simple')",
        "ev_tstr = UtilityEvaluator(",
        "    task=options_task, protocol='tstr',",
        "    num_epochs=8, batch_size=32, learning_rate=1e-3, verbose=False,",
        ")",
        "tstr_result = ev_tstr.run(real_train, real_test, syn_train)",
        "print('Protocol:', tstr_result['protocol'], '| Task:', tstr_result['task'])",
        "print('Useful (s_g > s_r):', tstr_result['useful'], '| delta =', f\"{tstr_result['score_delta']:+.4f}\")",
    ))

    cells.append(md(
        "### Per-channel paper-style table (Table 1)",
        "",
        "Each **row** is one channel × training source. Each column is the ",
        "mean ± std of one of the five PnL metrics over the 32 test ",
        "windows.",
    ))

    cells.append(code(
        "def fmt(agg, key):",
        "    return f\"{agg[key]['mean']:+.3f} ± {agg[key]['std']:.3f}\"",
        "",
        "rows = []",
        "for c in range(N_CHANNELS):",
        "    real_agg = tstr_result['real']['per_channel'][c]",
        "    syn_agg  = tstr_result['synthetic']['per_channel'][c]",
        "    rows.append({",
        "        'channel': c, 'source': 'Real (M_r)',",
        "        'U1 PnL': fmt(real_agg, 'pnl'),",
        "        'U2 Sharpe': fmt(real_agg, 'sharpe'),",
        "        'U3 CVaR α=0.05': fmt(real_agg, 'cvar'),",
        "        'U4 Win rate': fmt(real_agg, 'win_rate'),",
        "        'U5 Max DD': fmt(real_agg, 'max_drawdown'),",
        "    })",
        "    rows.append({",
        "        'channel': c, 'source': 'Synthetic (M̂_g)',",
        "        'U1 PnL': fmt(syn_agg, 'pnl'),",
        "        'U2 Sharpe': fmt(syn_agg, 'sharpe'),",
        "        'U3 CVaR α=0.05': fmt(syn_agg, 'cvar'),",
        "        'U4 Win rate': fmt(syn_agg, 'win_rate'),",
        "        'U5 Max DD': fmt(syn_agg, 'max_drawdown'),",
        "    })",
        "df1 = pd.DataFrame(rows)",
        "df1.style.set_caption(",
        "    \"Table 1 — Options task, per-channel U1–U5 (32 test windows, mean ± std)\"",
        ").format(precision=3)",
    ))

    cells.append(md(
        "### Aggregate across channels (Table 2)",
        "",
        "Per the paper, U1–U5 are aggregated as mean ± std across the N test ",
        "windows (eq. 27). Across channels we report the mean of channel ",
        "means and the std of channel means, so the reader can see both the ",
        "central tendency and the dispersion.",
    ))

    cells.append(code(
        "real_agg = tstr_result['real_aggregate']",
        "syn_agg  = tstr_result['synthetic_aggregate']",
        "agg_rows = []",
        "for source, agg in [('Real (M_r)', real_agg), ('Synthetic (M̂_g)', syn_agg)]:",
        "    agg_rows.append({",
        "        'training source': source,",
        "        'U1 PnL': fmt(agg, 'pnl'),",
        "        'U2 Sharpe': fmt(agg, 'sharpe'),",
        "        'U3 CVaR α=0.05': fmt(agg, 'cvar'),",
        "        'U4 Win rate': fmt(agg, 'win_rate'),",
        "        'U5 Max DD': fmt(agg, 'max_drawdown'),",
        "    })",
        "df2 = pd.DataFrame(agg_rows).set_index('training source')",
        "df2.style.set_caption(",
        "    \"Table 2 — Options task, cross-channel U1–U5 (mean ± std)\"",
        ").format(precision=3)",
    ))

    # ---- Section: Augmented --------------------------------------------- #
    cells.append(md(
        "## 3. Augmented training protocol (§6.1.2)",
        "",
        "Augmented training extends TSTR by also training `M̃` on the **full ",
        "union** `D_train ∪ D̂_g` (no balancing) and scoring it on `D_test` ",
        "alongside `M_r` and `M̂_g`.",
    ))

    cells.append(code(
        "ev_aug = UtilityEvaluator(",
        "    task=options_task, protocol='augmented',",
        "    num_epochs=8, batch_size=32, learning_rate=1e-3, verbose=False,",
        ")",
        "aug_result = ev_aug.run(real_train, real_test, syn_train)",
        "",
        "def agg_per_channel(payload):",
        "    return MetricToolbox.aggregate_per_channel(payload['per_channel'])",
        "",
        "rows = []",
        "for label, key in [('Real (M_r)', 'real'),",
        "                   ('Synthetic (M̂_g)', 'synthetic'),",
        "                   ('Augmented (M̃, full union)', 'augmented')]:",
        "    ag = agg_per_channel(aug_result[key])",
        "    rows.append({",
        "        'training source': label,",
        "        'U1 PnL': fmt(ag, 'pnl'),",
        "        'U2 Sharpe': fmt(ag, 'sharpe'),",
        "        'U3 CVaR α=0.05': fmt(ag, 'cvar'),",
        "        'U4 Win rate': fmt(ag, 'win_rate'),",
        "        'U5 Max DD': fmt(ag, 'max_drawdown'),",
        "    })",
        "df3 = pd.DataFrame(rows).set_index('training source')",
        "df3.style.set_caption(",
        "    \"Table 3 — Options task, Augmented protocol (full-union training)\"",
        ").format(precision=3)",
    ))

    # ---- Section: BS baseline ------------------------------------------- #
    cells.append(md(
        "## 4. BS static-delta baseline (premium = 0)",
        "",
        "Per the user's spec, the Black–Scholes static-delta baseline is run ",
        "with **premium = 0**: there is no option here, just the static BS ",
        "delta hedge on the underlying. We compare it directly against the ",
        "trained LSTM over the same `D_test`.",
    ))

    cells.append(code(
        "results = []",
        "",
        "# BS-static baseline — no training",
        "bs = BSStaticDelta(seq_length=SEQ_LEN, K=1.0, sigma_annual=0.20, time_horizon_years=1.0)",
        "pnl_bs = options_task.predict_period_pnl(bs, real_test[:, :, 0])",
        "agg_bs = MetricToolbox.aggregate([MetricToolbox.compute(pnl_bs[i]) for i in range(pnl_bs.shape[0])])",
        "results.append({'policy': 'BS static (premium=0)',",
        "                **{f'U1 PnL': fmt(agg_bs, 'pnl'),",
        "                   'U2 Sharpe': fmt(agg_bs, 'sharpe'),",
        "                   'U3 CVaR α=0.05': fmt(agg_bs, 'cvar'),",
        "                   'U4 Win rate': fmt(agg_bs, 'win_rate'),",
        "                   'U5 Max DD': fmt(agg_bs, 'max_drawdown')}})",
        "",
        "# LSTM Moneyness — train on real_train channel 0 only",
        "lstm = options_task.build_policy()",
        "lstm.fit(options_task.prepare_training(real_train[:, :, 0]),",
        "         num_epochs=8, batch_size=32, verbose=False)",
        "pnl_lstm = options_task.predict_period_pnl(lstm, real_test[:, :, 0])",
        "agg_lstm = MetricToolbox.aggregate([MetricToolbox.compute(pnl_lstm[i]) for i in range(pnl_lstm.shape[0])])",
        "results.append({'policy': 'LSTM Moneyness (channel 0, trained on real_train)',",
        "                **{f'U1 PnL': fmt(agg_lstm, 'pnl'),",
        "                   'U2 Sharpe': fmt(agg_lstm, 'sharpe'),",
        "                   'U3 CVaR α=0.05': fmt(agg_lstm, 'cvar'),",
        "                   'U4 Win rate': fmt(agg_lstm, 'win_rate'),",
        "                   'U5 Max DD': fmt(agg_lstm, 'max_drawdown')}})",
        "",
        "df4 = pd.DataFrame(results).set_index('policy')",
        "df4.style.set_caption(",
        "    \"Table 4 — Channel 0 reference: BS-static baseline vs trained LSTM, scored on real_test\"",
        ").format(precision=3)",
    ))

    # ---- Section: PnL toggle -------------------------------------------- #
    cells.append(md(
        "## 5. PnL toggle — simple vs log returns",
        "",
        "Inputs come in as **log returns**, but per-period PnL can be computed ",
        "two ways:",
        "",
        "* `simple`: `Δ_price = S_t · (exp(r) − 1)` (discrete-time).",
        "* `log`:    `Δ_price = S_t · r` (continuous-time approximation).",
        "",
        "Both are honest dollar amounts; the switch is a configurability ",
        "flag on each task (`pnl_returns='simple'|'log'`).",
    ))

    cells.append(code(
        "positions = torch.ones(real_test.shape[0], SEQ_LEN)",
        "positions[:, -1] = 0.0   # no position on the final day",
        "",
        "prices = torch.empty(real_test.shape[0], SEQ_LEN)",
        "prices[:, 0]  = 1.0",
        "prices[:, 1:] = torch.exp(torch.cumsum(real_test[:, :, 0], dim=1))",
        "",
        "pnl_simple = per_period_pnl_from_positions(positions, prices, mode='simple')",
        "pnl_log    = per_period_pnl_from_positions(positions, prices, mode='log')",
        "",
        "rows = []",
        "rows.append({'mode': 'simple',",
        "             'per-window mean PnL': f\"{pnl_simple.mean().item():+.4f}\",",
        "             'per-window sum |Δ|:': f\"{pnl_simple.sum(dim=1).abs().mean().item():.4f}\"})",
        "rows.append({'mode': 'log',",
        "             'per-window mean PnL': f\"{pnl_log.mean().item():+.4f}\",",
        "             'per-window sum |Δ|:': f\"{pnl_log.sum(dim=1).abs().mean().item():.4f}\"})",
        "df5 = pd.DataFrame(rows).set_index('mode')",
        "df5.style.set_caption(",
        "    \"Table 5 — PnL convention toggle on a long-1-unit position across 32 test paths\"",
        ").format(precision=3)",
        "",
        "diff = (pnl_simple - pnl_log).abs()",
        "print(f\"max |simple − log| across all (window, period) cells: {diff.max().item():.4f}\")",
        "print(f\"mean |simple − log| per window: {diff.mean(dim=1).mean().item():.4f}\")",
    ))

    # ---- Section: Final summary table ------------------------------------ #
    cells.append(md(
        "## 6. Paper-style headline table (Table 6)",
        "",
        "The final table consolidates the headline U1–U5 numbers from each ",
        "protocol run on the same test set. The `useful?` column applies the ",
        "paper's TSTR rule: a generator wins on U1 if `score_synth > score_real`.",
    ))

    cells.append(code(
        "def agg_table_rows(label, agg):",
        "    return {'run': label,",
        "            'U1 PnL': fmt(agg, 'pnl'),",
        "            'U2 Sharpe': fmt(agg, 'sharpe'),",
        "            'U3 CVaR α=0.05': fmt(agg, 'cvar'),",
        "            'U4 Win rate': fmt(agg, 'win_rate'),",
        "            'U5 Max DD': fmt(agg, 'max_drawdown')}",
        "",
        "rows = []",
        "rows.append(agg_table_rows('TSTR · real (M_r)', tstr_result['real_aggregate']))",
        "rows.append(agg_table_rows('TSTR · synth (M̂_g)', tstr_result['synthetic_aggregate']))",
        "rows.append(agg_table_rows('Augmented · real (M_r)', agg_per_channel(aug_result['real'])))",
        "rows.append(agg_table_rows('Augmented · synth (M̂_g)', agg_per_channel(aug_result['synthetic'])))",
        "rows.append(agg_table_rows('Augmented · full union (M̃)', agg_per_channel(aug_result['augmented'])))",
        "rows.append(agg_table_rows('BS-static baseline (premium=0)', agg_bs))",
        "rows.append(agg_table_rows('LSTM Moneyness (real, ch0)', agg_lstm))",
        "",
        "df6 = pd.DataFrame(rows).set_index('run')",
        "df6.style.set_caption(",
        "    \"Table 6 — StonkBench §6 utility on synthetic fake data (headline)\"",
        ").format(precision=3)",
    ))

    # ---- Closing -------------------------------------------------------- #
    cells.append(md(
        "## 7. Conclusions",
        "",
        "* **TSTR (§6.1.1)** produces a real-vs-synth comparison per ",
        "  generator. With random data the two scores are within noise and ",
        "  the `useful` flag flips covariantly with the seed.",
        "* **Augmented (§6.1.2)** adds `M̃` from full-union training without ",
        "  any balancing — paper-fidelity preserved.",
        "* **U1–U5 (§6.1.3)** are aggregated as mean ± std across the N test ",
        "  windows, exactly the paper's eq. 27.",
        "* **PnL convention** toggles cleanly between simple and log returns ",
        "  via a single flag.",
        "",
        "With real data the headline table becomes a paper-quality ",
        "generator-quality leaderboard.",
        "",
        "Next step: wire this orchestrator into `src/unified_evaluator.py` so ",
        "each generated artifact (one per `seq_length × model`) is scored ",
        "automatically and its U1–U5 JSON lands next to the fidelity / ",
        "diversity / stylized-facts metrics.",
    ))

    return cells


# --------------------------------------------------------------------------- #
# Notebook assembly                                                            #
# --------------------------------------------------------------------------- #
def build_notebook_dict() -> dict:
    return {
        "cells": build_cells(),
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


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook_dict()
    OUTPUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Wrote {OUTPUT}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
