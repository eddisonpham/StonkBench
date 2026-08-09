# §6 Utility Pipeline — Handoff

**Package:** `src/utility/`
**Paper sections implemented:** §6.1.1 TSTR · §6.1.2 Augmented · §6.1.3 U1–U5 toolbox · §6.2 Options · §6.3 Portfolio · §6.4 Alpha
**Status:** All three paper tasks operational, paper-faithful, smoke + adversarial validated.
**Tests:** 9/9 §6 smoke tests pass · 23/23 adversarial probes pass

---

## TL;DR

A modular **task × policy × protocol** pipeline that benchmarks synthetic financial-data generators end-to-end through three downstream tasks and reports the paper's U1–U5 PnL toolbox, aggregated as mean ± std across the N held-out test windows (eq. 27).

```
                       ┌──────────────────────────────────────────┐
                       │     UtilityEvaluator (orchestrator)      │
                       │   protocol="tstr" | "augmented"           │
                       └─────────────────────┬────────────────────┘
                                             │
                            ┌────────────────┼────────────────┐
                            ▼                ▼                ▼
                       TSTRProtocol     AugmentedProtocol   (per-window U1–U5)
                       §6.1.1           §6.1.2
                       M_r vs M̂_g       M_r vs M̂_g vs M̃
                            │                │
                            └────────┬───────┘
                                     ▼
                          ┌─────────────────────┐
                          │    BaseUtilityTask  │
                          │  Options (§6.2)     │
                          │  Portfolio (§6.3)   │
                          │  Alpha (§6.4)       │
                          └─────────┬───────────┘
                                    │ prepare_training + predict_period_pnl
                                    ▼
                          ┌─────────────────────┐
                          │   Policy module     │
                          │  MoneynessLSTM      │  ← per channel (Options §6.2.3)
                          │  PortfolioLSTM      │  ← whole window (§6.3)
                          │  AlphaLSTM          │  ← whole window (§6.4)
                          │  BSStaticDelta      │  ← Options baseline, premium=0
                          └─────────┬───────────┘
                                    ▼
                          ┌─────────────────────┐
                          │ MetricToolbox        │
                          │  U1 PnL · U2 Sharpe  │
                          │  U3 CVaR · U4 Win    │
                          │  U5 Max DD           │
                          │  → mean ± std (eq.27)│
                          └──────────────────────┘
```

The package is deliberately small (12 .py files, ~1.1 kLOC). All three tasks share the same metrics + protocols; only the task + policy module is task-specific.

---

## File inventory

### Core (every task uses these)

| File | Purpose |
|---|---|
| `metrics.py` | `MetricToolbox.compute(period_pnl)` → per-window U1–U5 dict. `MetricToolbox.aggregate(window_metrics)` → mean±std across N windows. `MetricToolbox.aggregate_per_channel(channel_aggregates)` → cross-channel collapse. `per_period_pnl_from_positions(positions, prices, mode)` → `(N, L−1)` per-period PnL — toggles between simple (`S_t·(exp r − 1)`) and log (`S_t·r`) returns. `safe_exp_cumsum(log_returns, dim)` — clamp-guarded `exp(cumsum)` helper, the single source of "clamp cumsum → exp" semantics so adversarial huge log returns don't propagate inf/NaN into the U1–U5 toolbox. |
| `protocols.py` | `TSTRProtocol.run(...)` (§6.1.1) — trains `M_r` on real, `M̂_g` on synthetic, scores both on real test; exposes `useful` and `score_delta` reflecting "generator useful iff `s_g > s_r`". `AugmentedProtocol.run(...)` (§6.1.2) — same plus `M̃` on the **full union** `D_train ∪ D̂_g` (no balancing); exposes `useful` + `score_delta` (TSTR) and `useful_aug` + `score_delta_aug` (paper §6.1.2's M̃ vs M_r comparison). Returns preserve `_full` payloads for diagnostics where callers want the raw per-channel list. |
| `evaluator.py` | `UtilityEvaluator(task, protocol="tstr"\|"augmented", num_epochs, batch_size, learning_rate, verbose)` — one-line orchestrator. Defaults are sane; everything is overridable. |
| `__init__.py` | Public re-exports so notebooks and downstream callers do `from src.utility import OptionsTask, …`. |

### Tasks — one per paper section

| File | Paper § | Shape | is_per_channel | Per-period PnL convention |
|---|---|---|---|---|
| `tasks/options.py` | §6.2 | `(N, L, C)` log returns | **yes** — one model per channel | `Δ_t · (S_{t+1} − S_t)` from the LSTM's `(Δ_{t-1}, g̃(t), τ_t)` autoregressive input |
| `tasks/portfolio.py` | §6.3 | `(N, L, I=20 inventory + J=5 hedge ETFs)` log returns | no, single whole-window model | `Σ_i π_{t,i} · (S_{t+1,i} − S_{t,i})` summed over I+J channels; `P_{t,i}=1` static unit inventory; honours user-supplied `channel_map` |
| `tasks/alpha.py` | §6.4 | `(N, L, I=25)` log returns (20 stocks + 5 ETFs) | no, single whole-window model | `Σ_i w_t,i · (S_{t+1,i} − S_{t,i})` summed over 25 assets with cross-sectional weights `w` |
| `tasks/base.py` | — | ABC | — | Defines the `prepare_training / build_policy / predict_period_pnl / extras` contract every task concrete class implements. |

### Policies — one paper-faithful model per task + static BS baseline

| File | Task | Loss | Architecture |
|---|---|---|---|
| `policies/lstm_moneyness.py` | Options (§6.2) | Quadratic replication error in moneyness units: `L_Δ = mean((g̃(L) − c₀/K − Σ_t Δ_t · (g̃_{t+1} − g̃_t))²)` with c₀ from Black–Scholes–Merton (r=0, σ annualised from training window) | Autoregressive LSTM, input `(Δ_{t-1}, g̃(t), τ_t)`. τ\_t = `1 − t/seq_length` exactly per paper §6.2 (clamped ≥ 0). Last Δ is unused for PnL (no return after maturity). |
| `policies/portfolio_lstm.py` | Portfolio (§6.3) | `mean_batch(Σ_t \|E_t\|)` where `E_t = I·P + Σ_j π_{t,j}` with `P_{t,i}=1` | Multivariate LSTM; whole `(N, L, I+J)` window feed; `hedge_clip=10` default bounds LSTM positions; leaves the inventory ones at 1.0. Loss has `torch.where` NaN/Inf guard. |
| `policies/alpha_lstm.py` | Alpha (§6.4) | Negative differentiable Sharpe ratio: `−mean_batch(μ/σ)` on per-window portfolio returns | Cross-sectional LSTM with `long_only=True` softmax (sums to 1) or `long_only=False` tanh (in [−1, +1]). `sharpe_eps=1e-4` floor on σ, μ\_safe guard for degenerate windows. |
| `policies/bs_static.py` | Options (baseline only) | none — analytic | Static Black–Scholes delta hedge, **premium=0** per user spec. Strict K-shape validation: raises ValueError on K of unexpected length. |
| `policies/base.py` | — | — | ABC defining `fit / predict / save / load / parameters_state`. |

---

## API quickstart

```python
from src.utility import (
    OptionsTask, PortfolioTask, AlphaTask,
    MoneynessLSTM, BSStaticDelta, PortfolioLSTM, AlphaLSTM,
    MetricToolbox, TSTRProtocol, AugmentedProtocol, UtilityEvaluator,
)
import torch

# Data: log returns tensors of shape matching the paper's
# rolling-window convention (N windows × L trading days × C channels).
real_train     = torch.randn(64, 251, 25) * 0.01   # (N_train, L, C)
real_test      = torch.randn(16, 251, 25) * 0.01   # (N_test,  L, C)
synthetic_train = torch.randn(64, 251, 25) * 0.01
init_dollars    = torch.ones(16)                   # (N_test,) per-window $

# Choose task + protocol:
task = PortfolioTask(num_inventory=20, num_hedge=5, seq_length=251)
ev   = UtilityEvaluator(task=task, protocol="tstr",
                        num_epochs=8, batch_size=16, learning_rate=1e-3)

result = ev.run(real_train, real_test, synthetic_train,
                real_test_initial_dollars=init_dollars)

print(result["protocol"], result["task"])         # "tstr" "portfolio"
print(result["useful"], result["score_delta"])    # True / +0.064
print(result["real"]["pnl"])                      # {"mean": 1.06, "std": 0.91}
```

For Options:
```python
task = OptionsTask(seq_length=252, premium_mode="bs", pnl_returns="simple")
real_train    = torch.randn(64, 252, 3) * 0.01   # C=3 channels
real_test     = torch.randn(16, 252, 3) * 0.01
synthetic_train = torch.randn(64, 252, 3) * 0.01
result = UtilityEvaluator(task=task, protocol="augmented").run(
    real_train, real_test, synthetic_train,
)
# result["real_aggregate"], result["synthetic_aggregate"],
# result["useful"], result["score_delta"],
# result["useful_aug"], result["score_delta_aug"]
```

For Alpha:
```python
task = AlphaTask(num_assets=25, seq_length=252, long_only=True)
result = UtilityEvaluator(task=task, protocol="tstr").run(
    real_train, real_test, synthetic_train,
)
```

**Toggle PnL convention** per task via `pnl_returns="simple"|"log"`:
- `simple`: `Δ_price = S_t · (exp(r) − 1)` — discrete-time, paper-faithful.
- `log`: `Δ_price = S_t · r` — continuous-time approximation. Both produce dollar amounts.

**Per-window dollar scaling** — every task's `predict_period_pnl(..., initial_dollars=None)` scales linearly (defaults to $1 for every window). Pass a `(N,)` tensor or `None`; raises ValueError on shape mismatch (silent expand was a footgun we removed).

---

## Paper-fidelity map (§6.X → file:line)

| Paper says | Landed here |
|---|---|
| §6.1.1 TSTR on `D_train`/`D̂_g`, score on `D_test`, `useful iff s_g > s_r` | `protocols.TSTRProtocol.run` — trains both, scores both, exposes `useful` & `score_delta` |
| §6.1.2 Augmented on `D_train ∪ D̂_g` (full union, no balancing) | `protocols.AugmentedProtocol._union` = `torch.cat([a, b], dim=0)`. Plus `useful_aug` & `score_delta_aug` for the M̃ vs M_r comparison. |
| §6.1.3 U1–U5 toolbox, mean ± std across N (eq. 27) | `metrics.MetricToolbox.{compute, aggregate, aggregate_per_channel}` |
| §6.1.3 U3 CVaR = mean of worst α-tail of per-period PnL | `MetricToolbox.compute` — `sorted_pnl[:ceil(α·T)].mean()` |
| §6.1.3 U2 Sharpe w/o annualisation (paper eq. 27) | `mean/std` of per-period PnL; no √L (single-year horizon) |
| §6.1.3 per-period PnL = `Σ_i π_{t,i} r_{t+1,i}` (simple) | `metrics.per_period_pnl_from_positions` |
| §6.2 European call, `(S_L − K)⁺` payoff, BS-Merton premium `c₀` | `policies.lstm_moneyness.bs_call_price(S=1, K, σ_annualised, T=1, r=0)` |
| §6.2 Moneyness `g̃(t) = S_t/K` | `_reconstruct_prices` + `MoneynessLSTM.predict` |
| §6.2 LSTM input `(Δ_{t-1}, g̃(t), τ_t)` | `MoneynessLSTM.predict`'s per-step `torch.stack` |
| §6.2 τ\_t = `1 − t/L` exactly | `MoneynessLSTM.predict` — `denom = self.seq_length`, clamped ≥ 0 |
| §6.2 7-moneyness augmentation `g̃(0) ∈ {0.7…1.3}` | `tasks.options.DEFAULT_MONEYNESS_GRID` exercised in `OptionsTask.prepare_training` |
| §6.2 Loss = quadratic replication error in moneyness units | `MoneynessLSTM.loss` |
| §6.2 BS-static baseline | `policies.bs_static.BSStaticDelta` with **premium=0** per user spec |
| §6.3 Portfolio I=20 inventory, J=5 ETFs, static unit inventory `P_t=1` | `tasks.portfolio.PortfolioTask._assemble_full_pos` with `inventory_unit=1.0` |
| §6.3 Loss `Σ_t \|E_t\|` | `policies.portfolio_lstm.PortfolioLSTM.loss` |
| §6.3 Whole multi-asset `(L, I+J)` fed to LSTM | `PortfolioLSTM.predict` consumes `windows[:, :-1, :]` (L−1 price steps) and emits L−1 hedge positions, pads the last to 0 |
| §6.4 I=25 assets, cross-sectional weights `π ∈ ℝᴵ` | `policies.alpha_lstm.AlphaLSTM` with `softmax` (long\_only) / `tanh` (short) |
| §6.4 Loss = negative differentiable Sharpe | `AlphaLSTM.loss` = `−mean_batch(μ/σ)`, `sharpe_eps` floor on σ, μ\_safe guard for degenerate windows |

---

## Validation harness

Two layers of regression coverage; run both before merging any §6 refactor.

```bash
# 9 functional end-to-end tests (Options TSTR, Augmented, BS-baseline,
# PnL toggle, Portfolio/Alpha smoke, per-paper TSTR, initial_dollar,
# channel_map inversion).
venv/bin/python scripts/smoke_utility_pipeline.py

# 23 hostile-input probes that defend the boundary contracts:
#   - N=0 / N=1 windows  - log-returns direct to policies
#   - K / initial_dollars of wrong/negative size  - channel_map inversion
#   - degenerate Sharpe windows  - extreme log-return magnitudes
#   - mismatched L between train/test  - augment union math
venv/bin/python scripts/adversarial_utility.py
```

Both should end with `All §6 utility smoke tests passed.` and `PASSES: 23/23` respectively. The full overview of contract behaviour is in `scripts/adversarial_utility.py` — the file is the spec.

---

## Notebooks

| Notebook | Task | Verified commands |
|---|---|---|
| `notebooks/01_utility_pipeline_smoke.ipynb` | Options (§6.2) | Sections 1–7 cover synthetic data → TSTR Tables 1–2 → Augmented Table 3 → BS-static baseline Table 4 → PnL toggle Table 5 → headline Table 6 → conclusions |
| `notebooks/02_portfolio_hedge_smoke.ipynb` | Portfolio (§6.3) | TSTR Table 1 → Augmented Table 2 → `channel_map` inversion regression → `initial_dollar` per-window scaling |
| `notebooks/03_alpha_gen_smoke.ipynb` | Alpha (§6.4) | TSTR Table 1 → Sharpe diagnostic → Augmented Table 2 → cross-sectional weight sum check → `long_only` vs long-short Table 3 → `initial_dollar` per-window scaling |

All three notebooks regenerate from JSON-emitting build scripts so source-of-truth is one place:

```bash
venv/bin/python scripts/build_smoke_notebook.py     # → 01_...
venv/bin/python scripts/build_extra_notebooks.py    # → 02_…  03_…
venv/bin/jupyter lab notebooks/01_utility_pipeline_smoke.ipynb
```

`scripts/execute_notebook.py` runs notebook 01 end-to-end via stdlib `exec()` (no `nbformat`/`ipykernel` dependency) and writes outputs back to the `.ipynb`. Useful for CI smoke-runs without installing Jupyter.

---

## Bug history — adversarial audit trail

This is the deliberate-flaw → fix → regression-test timeline. Each row was caught by an adversarial probe in `scripts/adversarial_utility.py`; the probe stays in the suite so the bug cannot silently regress.

| # | Where | Bug | Fix |
|---|---|---|---|
| 1 | `lstm_moneyness.predict` log-returns branch | Dead `K * None` placeholder threw `TypeError` whenever a caller passed `(N, L-1)` log returns directly | Cleaned up the branch; S0 anchored at K, prices reconstructed via `safe_exp_cumsum`. |
| 2 | `_reshape_init_dollars` (protocols) | `init.expand(N)` silently crashed on `[1, 5]` against `N=3` (size mismatch) | Strict `ValueError` with length context; size-1 broadcast still allowed. |
| 3 | `tasks/portfolio.py`, `tasks/alpha.py`, `tasks/options.py` | Same silent-expand pattern in three more places | Mirrored the strict ValueError; default $1 fallback documented. |
| 4 | `bs_static.predict` K-shape | `bs_delta_step` silently crashed inside BS Δ call when K shape was neither (1,) nor (N,) | Added the same strict K-shape check. |
| 5 | `lstm_moneyness.predict` K-shape | Same silent K broadcast — inconsistent with the new BS-static contract | Mirrored the strict shape check across both policies. |
| 6 | `OptionsTask.predict_period_pnl` | Accept-but-ignored `initial_dollars` argument (inconsistent with `PortfolioTask`/`AlphaTask`) | Now scales `pnl_unit * init.unsqueeze(-1)` everywhere, default `None` → `$1` for every window. |
| 7 | `tasks/portfolio.py & tasks/alpha.py prepare_training` | Adversarial huge log returns overflowed float32 in `exp(cumsum)`, producing `inf/NaN` prices | DRY'd into `safe_exp_cumsum(log_returns, dim)` helper, called from 5 sites (Portfolio, Alpha, Options 2D, Options predict-reconstruct, BS-static). |
| 8 | `OptionsTask._reconstruct_prices` & `_moneyness.predict` | Same overflow — never caught because probes only exercised 3D paths | Same `safe_exp_cumsum` helper. |
| 9 | `lstm_moneyness.predict` τ formula | Used `1 − t / prices.shape[1]` — off by one in the final maturity step (paper §6.2 exact: `1 − t/L`) | Use `1 − t / self.seq_length` exactly, clamped ≥ 0. |
| 10 | `AugmentedProtocol.run` | No `useful_aug` flag — paper §6.1.2's "M̃ vs M_r" comparison wasn't computable from the result dict | Added `useful_aug` + `score_delta_aug` (plus a helper `_topagg()` collapses per-channel results so the keys are consistent across task types). |
| 11 | `AugmentedProtocol.run` post-fix | `_best_generator` is a `@staticmethod` on `TSTRProtocol`; calling `self._best_generator(...)` from `AugmentedProtocol` raised `AttributeError` | Call it via `TSTRProtocol._best_generator(...)`. |
| 12 | `notebooks/02_portfolio_hedge_smoke.ipynb` augmented cell | `MetricToolbox.aggregate(result[key])` iterated `{aggregate, extras}` dict keys as strings, filtering everything out → all-NaN U1–U5 | Drilling helper that handles BOTH `{per_channel, extras}` (per-channel tasks) and `{aggregate, extras}` (whole-window tasks) shapes. Same fix in `notebooks/03_alpha_gen_smoke.ipynb`. Same fix mirrored in `scripts/build_extra_notebooks.py` so a regen produces correct cells. |

The probes (`scripts/adversarial_utility.py`) cover cases 1–11 explicitly; case 12 is the same root cause as case 10 spotted during interactive notebook review.

---

## Known limitations / open paper-spec gaps

| Topic | Paper § | Status |
|---|---|---|
| Hyperparameter rolling-window validation | §6.1 (intro) | **Not implemented.** Callers pass `num_epochs`, `batch_size`, `learning_rate` directly. Worth a follow-up: add a fold-aware grid search across (epochs, batch, lr, hidden) per task and pipe the chosen config into final training. |
| Per-channel cross-task hyperparameters | — | All channels share the same `num_epochs/batch_size/learning_rate`. Acceptable for smoke; a real run would either per-channel-tune or share weights via `module = nn.Module` factoring. |
| Numerical precision (float32) | — | All pipelines run in float32. `safe_exp_cumsum` caps at ±40; can be widened for long horizons (>1 trading year) by changing `CUM_SUM_CLAMP`. |
| Per-period PnL boundary | — | Positions of shape `(N, L)` are indexed `0..L−2` against prices `1..L−1`, producing `L−1` per-period PnL entries. That matches the paper's "L trading periods" convention for a window of `L` returns, but callers wanting exactly `L` should re-anchor. |
| BS-static baseline scope | §6.2 | Only used for Options; Portfolio/Alpha don't have an analytic baseline. To build one for these, add `policies/portfolio_static.py` / `policies/alpha_static.py` similar to `bs_static.py`. |
| `AugmentedProtocol` return shape | — | Top-level `result["real"]` holds the raw `_fit_score` payload (shape depends on `is_per_channel`). For safer notebook usage the cells now drill via a helper; we could standardise the top-level shape to always be a collapsed aggregate dict, but it's a backwards-compat trade-off. |
| Long-short `tanh` config for Alpha | §6.4 | Available as `AlphaTask(long_only=False)` but the paper itself doesn't specify; keep both, document. |
| Top-of-book per-window interpretability | — | `delta_price` aggregator drops the per-asset breakdown into a scalar `PnL_t`. For Plot Statistics the per-asset contribution is in `tasks/portfolio.py.predict_period_pnl`'s `(held * delta_price)` pre-sum — caller can request that intermediate if needed. |

---

## Extension recipe — adding a new task

Following the existing pattern, a new downstream task is four changes:

```python
# src/utility/tasks/my_new_task.py
class MyNewTask(BaseUtilityTask):
    task_name = "my_new"
    policy_family = "my_policy"
    is_per_channel = False                # set True for per-channel training

    def __init__(self, ..., pnl_returns="simple", default_initial_dollar=1.0):
        super().__init__(pnl_returns=pnl_returns)
        ...

    def prepare_training(self, log_returns):            # → task-specific training payload
        ...

    def build_policy(self):                              # → fresh, untrained nn.Module + BaseUtilityPolicy
        return MyNewPolicy(...)

    def predict_period_pnl(self, policy, test_windows, initial_dollars=None, **_):
        positions = policy.predict(prices)
        pnl_unit = per_period_pnl_from_positions(positions, prices, mode=self.pnl_returns)
        return pnl_unit * init.unsqueeze(-1)             # honour initial_dollars (None → $1 default)

    def extras(self, policy, test_windows):              # optional task-specific diagnostics
        ...

    def run_one(self, train_windows, test_windows, initial_dollars=None, *, ...):  # for whole-window
        # build → fit → predict
```

Then:

1. Create `src/utility/policies/my_new_policy.py` — subclass both `nn.Module` AND `BaseUtilityPolicy`. Implement `fit`/`predict` plus your task-specific loss.
2. Re-export from `src/utility/__init__.py`.
3. Add a smoke test in `scripts/smoke_utility_pipeline.py` and an adversarial probe in `scripts/adversarial_utility.py`.
4. Add a notebook (or a new section to an existing one) via `scripts/build_extra_notebooks.py`.

The existing `TSTRProtocol` and `AugmentedProtocol` will pick up your task automatically as long as you set `is_per_channel` correctly and `run_one_channel` exists if it's per-channel.

---

## Known interaction shape gotcha

`AugmentedProtocol.run()` and `TSTRProtocol.run()` return **different shapes** for per-channel vs whole-window tasks:

| Task type | `TSTRProtocol.run()["real"]` | `AugmentedProtocol.run()["real"]` |
|---|---|---|
| **Per-channel** (Options) | `{per_channel: [...], extras: [...]}` (raw fit\_score) | same as TSTR |
| **Whole-window** (Portfolio/Alpha) | **collapsed aggregate** (`{pnl: {mean, std}, sharpe: {...}, ...}`) | **raw fit\_score** (`{aggregate: {...}, extras: {...}}`) |

Use the notebooks' `_topagg()` helper when comparing TSTR/Augmented tables side-by-side:

```python
def _topagg(payload):
    if "per_channel" in payload:
        return MetricToolbox.aggregate_per_channel(payload["per_channel"])
    if "aggregate" in payload:
        return payload["aggregate"]
    return MetricToolbox.aggregate([payload])
```

Returning the same shape across both protocols is the obvious refactor — flagged as a known limitation above.

---

## Where the original review and bug report live

- `/Users/uyenlamho/Downloads/UTILITY_REVIEW.md` — original audit enumerating the 8 protocol deviations from paper §6 in the *preceding* implementation. Every item has been fixed in this package.
- `/Users/uyenlamho/Downloads/stonk-bench.pdf` — paper source of truth (PDF; `pdfplumber`/`pdftotext` not installed in this venv, so section quotations in this MD taken from UTILITY\_REVIEW\.md).
- `scripts/smoke_utility_pipeline.py` — paper-fidelity happy-path tests.
- `scripts/adversarial_utility.py` — boundary-condition regressions for the bugs above.

---

## Closing notes for the next contributor

- **Don't change the protocol return shapes without a deprecation cycle.** The cross-task shape asymmetry (per-channel vs whole-window) is the documented gotcha; flattening it is a real refactor that should drop the `_topagg()` helpers and update all three notebooks.
- **Keep `safe_exp_cumsum` as the single source of `exp(cumsum)` semantics** in the package. Any new `prepare_training`/`predict` that reconstructs prices from log returns should call it; raw `torch.cumsum(...) → torch.exp(...)` chains have proven to leak inf/NaN under adversarial inputs.
- **Add a smoke test + an adversarial probe** for every new bug fix. The probes are cheap, the table is dense, and the bug history above is the cheapest living documentation we have.
- **The four validation commands are the contract** — `smoke_utility_pipeline.py`, `adversarial_utility.py`, `build_extra_notebooks.py`, `execute_notebook.py`. Run them all before sending a PR that touches `src/utility/`.
