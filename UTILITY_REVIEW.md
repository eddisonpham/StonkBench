# Utility Implementation vs. Paper §6 — Review

**Date:** August 8, 2026
**Scope:** `src/taxonomies/utility.py`, the `UtilityEvaluator` wrapper in `src/utils/evaluation_classes_utils.py`, hedger internals in `src/hedging_models/`, and how they are invoked from `src/unified_evaluator.py`.
**Reference:** *StonkBench: Unified Benchmark for Synthetic Market Generation for Historical Financial Time Series* (Ho & Pham), Section 6 — "Utility Evaluation of Downstream Tasks".

---

## 1. What the paper (§6) specifies

Section 6 defines a **train-synthetic-test-real (TSTR)** utility evaluation built from **three downstream tasks** scored with a **five-metric PnL toolbox**.

### 1.1 Common protocol (§6.1)
- Each task T is specified by a loss `L_T(θ)` over a policy θ; the optimal policy is `θ* = argmin L_T(θ)`.
- Policies are trained under TSTR (§6.1.1) on the data partition of §4.1 with **rolling-window validation** for hyperparameter tuning.
- **§6.1.1 TSTR:** training algorithm A on `D_train` and `D̂_g` yields `M_r` and `M̂_g`. Both are scored on the **real test set** `D_test`, giving `s_r` and `s_g`. The generator is useful for pair (T, A) iff `s_g > s_r`; the best generator is `G_g*` with `g* = max{s_g}`.
- **§6.1.2 Augmented training:** additionally train on `D̃_j = D_train ∪ D̂_j` (full union), yielding `M̃_j`, to test whether synthetic data adds value on top of real training data.
- **§6.1.3 Metric toolbox (U1–U5):** given portfolio `Π_M = (π_{t,i}) ∈ R^{L×I}` with per-period PnL `PnL_t(Π_M) = Σ_i π_{t,i} r_{t+1,i}` (simple returns):
  - **[U1] PnL** — total PnL over the window.
  - **[U2] Annualized Sharpe** — mean per-period PnL / std. *No annualization factor (√L) is applied* (single trading-year horizon).
  - **[U3] CVaR** — expected loss in the worst α-tail of per-period PnL.
  - **[U4] Win rate** — fraction of periods with positive PnL.
  - **[U5] Max drawdown** — largest peak-to-trough decline of cumulative PnL.
- Each metric is reported as **mean ± std across all N test windows** (eq. 27).

### 1.2 The three tasks
- **§6.2 Option delta hedging:** European call at strike K, maturity L, payoff `(S_{L,i} − K)⁺`, rebalanced end-of-day. The loss is the **quadratic replication error in moneyness units**: `g̃(t) = S_{t,i}/K`, `g̃(L) = (S_{L,i}/K − 1)⁺`, `g′(L) = g̃(L) − c₀/K` where c₀ is the Black–Scholes–Merton premium. Framework: **LSTM** fed the triplet `(Δ_{t−1}, g̃(t), τ_t)` with normalized time-to-maturity `τ_t = 1 − t/L`; one asset per channel. Training augments each window with **7 initial moneynesses `g̃(0) ∈ {0.7, …, 1.3}`** (N×7 examples), propagating log-moneyness recursively (`log g̃(t) = log g̃(t−1) + R_{t,i}`). Evaluation reports the U1–U5 toolbox **plus** the hedging loss L_Δ.
- **§6.3 Portfolio hedging:** I = 20 inventory stocks hedged with J = 5 liquid ETFs, static unit inventory `P_t = 1`, same LSTM framework, loss = cumulative absolute net exposure.
- **§6.4 Alpha generation:** I = 25 assets (20 stocks + 5 ETFs), LSTM with cross-sectional output, loss = **negative differentiable Sharpe ratio**.

---

## 2. How the current implementation strays

### 2.1 Structural gaps (largest)

| # | Deviation | Paper §6 says | Current code does |
|---|-----------|---------------|-------------------|
| 1 | **U1–U5 toolbox missing** | Score everything with PnL, Sharpe (no √L), CVaR, win rate, max drawdown, aggregated mean±std over test windows (eq. 27). Hedging loss is an *additional* report (§6.2.4). | Only reports `mean`/`std` of the replication error R (`summarize_replication_error`). None of U1–U5 are computed. |
| 2 | **Two of three tasks missing** | Portfolio hedging (§6.3) and alpha generation (§6.4). | Not implemented at all. |
| 3 | **Spearman "algorithm comparison" not in the paper** | §6.1.1: per-generator comparison `s_g` vs `s_r`, useful iff `s_g > s_r`, best generator = `argmax s_g`. | `AlgorithmComparisonEvaluator` ranks hedgers on real vs synthetic scores and reports the Spearman correlation of the two rankings — an invented protocol. (The "train on synthetic, score on real test" half *does* match TSTR.) |

### 2.2 Protocol deviations in the option-hedging path that does exist

| # | Deviation | Paper §6 says | Current code does |
|---|-----------|---------------|-------------------|
| 4 | **No moneyness normalization, no BS premium** | Loss on moneyness units: `g̃(L) − c₀/K` with c₀ from Black–Scholes–Merton. | Loss on raw `(S_L − K)⁺` minus terminal value; premium is a **learned `nn.Parameter`** optimized by MSE (`base_hedger.py`), not c₀/K. |
| 5 | **Single ATM strike, no strike generalization** | Augment each window with 7 initial moneynesses `{0.7, …, 1.3}` (N×7 examples). | `strike = mean(initial prices)` ⇒ moneyness `g̃(0) = 1` always. |
| 6 | **Wrong network input/architecture** | LSTM fed `(Δ_{t−1}, g̃(t), τ_t)`; moneyness-based, one asset per channel. | Raw **price paths** are fed; no moneyness, no τ_t, no Δ feedback, no log-moneyness recursion. Extra hedgers (Feedforward, RNN, BlackScholes, DeltaGamma, LinearRegression, XGBoost) replace the paper's LSTM-only setup. |
| 7 | **Augmented training semantics** | `D̃_j = D_train ∪ D̂_j` (full union, no balancing), evaluated on **D_test**. | `AugmentedTestingEvaluator` uses a 50/50 **balanced subsample** (min of sizes) and evaluates on the real **validation** set; compares real-only vs mixed rather than the three-way `{M_r, M̂_g, M̃_j}`. |
| 8 | **No hyperparameter tuning** | Rolling-window validation for hyperparameters (§6.1). | Fixed `num_epochs` / `batch_size` / `learning_rate`. |

### 2.3 Minor
- Paper rebalances end-of-day with `Δ ∈ R^L`; code produces L−1 deltas (`prices[:, :-1]`) — essentially consistent, not worth changing.

---

## 3. What actually matches

- Training on real vs synthetic and scoring on the **real test set** (in `AlgorithmComparisonEvaluator`, synthetic-trained hedgers are scored on `real_test_prices`) — consistent with TSTR in spirit.
- The quadratic (MSE) form of the replication-error loss used during hedger training mirrors eq. 28's quadratic loss, modulo units.
- Hedging one asset at a time (univariate per-channel processing) matches §6.2.3.

---

## 4. Bottom line

If the goal is to match paper §6, the utility axis needs a rewrite:

1. Implement the **U1–U5 PnL toolbox** on the per-period PnL of the hedging portfolio, aggregated as mean±std over test windows.
2. Rewrite the option-hedging path per §6.2: **moneyness-based LSTM** with `(Δ_{t−1}, g̃(t), τ_t)` inputs, 7-strike augmentation, BS premium, and the replication-error loss in moneyness units.
3. Align training/evaluation with §6.1.1–6.1.2: `{M_r, M̂_g, M̃_j}` trained on `{D_train, D̂_g, D_train ∪ D̂_g}`, all scored on **D_test**, reporting `s_g` vs `s_r` per hedger.
4. Implement the two missing tasks: **portfolio hedging (§6.3)** and **alpha generation (§6.4)**.

---

## 5. Files referenced

- `src/taxonomies/utility.py` — `AugmentedTestingEvaluator`, `AlgorithmComparisonEvaluator`, replication-error helpers.
- `src/utils/evaluation_classes_utils.py` — `UtilityEvaluator` orchestration.
- `src/hedging_models/base_hedger.py` — learned premium, terminal value, MSE loss.
- `src/hedging_models/deep_hedgers/` — LSTM / Feedforward / RNN hedgers (raw-price inputs).
- `src/hedging_models/non_deep_hedgers/` — BlackScholes / DeltaGamma / LinearRegression / XGBoost.
- `src/unified_evaluator.py` — `UtilityMetricsEvaluator` glue (splits, initial prices, channel handling).
