# Known issues — corrected downstream evaluations, and what is still unfixed

Reproduce everything below with:

```bash
PYTHONPATH=<numpy+scipy site-packages> python3 scripts/handover/audit_downstream.py
```

Every claim is checked against the **shipped arrays** in this bundle, and the
pre-fix formulas reproduce the shipped numbers exactly (so the diagnosis is
about the code, not the data). The last section of the audit re-checks the
results JSON that now ships in `04_evaluation_results/`.

**Trigger:** a reviewer noticed `pnl.synthetic.sortino_ratio` reaching 6+ on one
model while everything else sat near zero. That symptom is real and it has three
separate causes, two of them outright bugs.

---

## 1. Defects found

| # | where | defect | severity |
|---|---|---|---|
| P1 | `portfolio._minimum_variance_weights(allow_short=False)` | SLSQP converges in 1 iteration **at its 1/N starting point**; the code accepted it. Long-only GMV == 1/N in **144/144** shipped strategy-size pairs. | high |
| P2 | `portfolio._evaluate_fixed_assets` | `_Real_GMV` and `_Real_GMV_LongOnly` (and both tangency variants) mapped to the **same** key. Only 2 of 4 real baselines existed, and they held the **long-only** values. | high |
| P3 | `portfolio._portfolio_metrics` | `turnover` only computed when a weight *history* exists; the evaluator always passes one row → **0.0 in 1008/1008** entries. | medium |
| P4 | `portfolio.PortfolioOptimizationEvaluator` | `estimation_window` / `rebalance_freq` are stored and documented but **never used**. No rolling re-estimation, no transaction costs — despite the docstring claiming DeMiguel et al. (2009) out-of-sample methodology. | high (methodology) |
| B1 | `pnl._pnl_metrics`, `portfolio._portfolio_metrics` | `cumprod(1+r)` / `prod(1+r)` applied to **log** returns, while `_accumulate_returns` (same output) uses `exp(cumsum(r))`. Two inconsistent terminal-wealth definitions — and the portfolio block's drawdown path had the same bug. | medium |
| B2 | `pnl._pnl_metrics` | Sortino denominator `sqrt(mean(min(0,r)²)) + 1e-15`. A path with no losses makes the denominator 0 and the epsilon turns "undefined" into a huge finite number — **6.65e12** shipped. Omega/tail silently dropped `inf` paths. | high |
| B3 | `unified_evaluator` (upstream) | the "N paths" of the real arm are **overlapping stride-1 windows** of one series (99.2 % overlap at L=126), averaged as if independent. | high (methodology) |
| B4 | `unified_evaluator` (upstream) | the real reference is truncated to `min(reference_len, n_samples)`, so the real arm's own value depends on which model is being evaluated (3 distinct "real" Sharpe values at L=21 for one identical series). | high (methodology) |
| B5 | `pnl` | `equal_weight_bh` is model-free, so `pnl.synthetic.*` describes the synthetic data's mean/vol ratio, not any decision quality. With `timegrad`'s synthetic vol at 0.069 vs the real 0.140, Sharpe/Sortino/Calmar inflate together. | informational |
| R1 | `unified_evaluator.DatasetCache._raw_series` | the *statistical* branch is labelled `statistical_test_windows`, but `_raw_series("test")` reads **`self._dl_set`**, not the statistical set. The 596-row `statsmodel_set.test_series` is never used for windows. This is why every model family's real pool is sized by the DL set's 260-row test split. | high (methodology) |
| D3 | reference sizing | consequence of R1: the stride-1 pool over a 260-row test series is 240 / 219 / 135 / **9** windows at L = 21 / 42 / 126 / 252, and the DL branch then takes its second half (120 / 110 / 68 / **5**). See §3. | high |

---

## 2. The Sortino symptom, explained

`timegrad`, L=21, synthetic arm (9 paths):

| quantity | value |
|---|---|
| shipped `sortino_ratio` | **6.65453e+12** |
| min downside RMS across paths | **0.0** (one path has no negative step at all) |
| post-fix `sortino_ratio` | **58.46**, with `n_sortino_undefined = 1`, `n_sortino_defined = 8` |

One of nine paths has zero downside deviation; `mean / (0 + 1e-15)` produced
~6e13 for that path and `np.mean` carried it into the reported value. The
`1e-15` epsilon did not protect against the division — it *created* the blow-up
by replacing "undefined" with an arbitrary huge number.

The residual 58 is **not** a formula problem: this model's synthetic returns have
roughly half the real volatility and a large positive drift, and the P&L
"strategy" is a fixed equal-weight buy-and-hold. The fix at least makes the
number finite, flagged and countable.

Other cells affected by the same epsilon: 28 entries across the 48 cells × 2 arms
exceeded |10| on Sortino/Calmar/Omega, and `kalman_vae_safe` reached
Sortino 12.45 / Calmar 32.7 at L=21. The **real** arm's `calmar_ratio` was also
affected (23.36 at L=21) because a 21-day window has a tiny drawdown.

---

## 3. Why L=252 carries almost no information

This is not a modelling defect — it is arithmetic on a short test split, made
worse by R1. From `data/preprocessed/dl_set_W252.pt`:

| | value |
|---|---|
| `test_series` (and `test_series_raw`) rows | **260** |
| `test_windows` at W=252 | **9** |

so at stride 1 the real pool is:

| L | 21 | 42 | 126 | 252 |
|---|---|---|---|---|
| statistical branch (whole pool) | 240 | 219 | 135 | **9** |
| DL branch (second half) | 120 | 110 | 68 | **5** |

The evaluator then aligns with `n = min(pool, n_samples)`, which is why the
effective sample counts are 240/219/135/9, 120/110/68/5 and 9/9/9/5.

So at **L=252 every comparison rests on 9 real windows (DL cells: 5)**, and for
the five 9-sample DL models only 5 synthetic samples survive the alignment. Any
metric that depends on the real set — fidelity distances, stylised facts,
terminal-wealth Wasserstein, the P&L gaps — is noise-dominated there. Diversity
needs no reference and is unaffected.

Two further consequences:

* **L=252 is the DL models' training length.** It is the only horizon where they
  are at their trained window, yet it is the horizon with the least real data.
* A fix would be to evaluate L=252 against a longer test split (or the
  statistical set's own `test_series`, which has 596 rows → 345 windows) and to
  keep the reference fixed across models. Nothing was regenerated here.

`2026-10-11`: `03_ground_truth/eval_reference/ground_truth_seq252.pt` also has
only 9 windows, so the ground truth shipped alongside the run is not the
limiting factor — the evaluator's pool is.

---

## 4. What was changed in the code

`src/taxonomies/pnl.py`

* wealth compounds as `exp(cumsum(log r))` and terminal wealth as
  `exp(sum(log r))`, matching `_accumulate_returns` used for the Wasserstein
  distance in the same output;
* the downside deviation is taken around a configurable MAR and compared to a
  *relative* tolerance; paths with no downside are recorded as `inf`/`nan` and
  **excluded from the average**, with `n_sortino_defined`,
  `n_sortino_undefined`, `n_omega_infinite` and `n_paths` added to the output;
* Omega returns `inf` when every path is infinite instead of `nan` from
  `np.mean([])`; the tail ratio no longer treats a zero P5 as `1e-12`.

`src/taxonomies/portfolio.py`

* the drawdown path compounds log returns as `exp(cumsum(r))`, matching
  `pnl`/`_accumulate_returns` (it used to use `cumprod(1 + r)`); a portfolio
  path with zero downside deviation reports an `inf`/`nan` Sortino instead of a
  misleading `0.0`;
* `_minimum_variance_weights(allow_short=False)` rescales the objective to O(1),
  optimises from several warm starts (1/N, inverse-variance, projected
  unconstrained GMV) and keeps the best feasible point; if it still cannot beat
  1/N it now **warns** and returns inverse-variance weights rather than silently
  returning 1/N. Verified: the objective drops from 1.26e-4 to 8.10e-5 at
  `block_bootstrap` L=126 C=25, and the fix beats 1/N in 15/15 probed cases;
* `_tangency_weights(allow_short=False)` gets the same multi-start treatment;
* the four real baselines now get **explicit distinct keys**
  (`Real_GT_RealGMV`, `Real_GT_RealGMV_LongOnly`, `Real_GT_RealTangency`,
  `Real_GT_RealTangency_LongOnly`). Note the semantic change: the plain
  `Real_GT_RealGMV` key now holds the *unconstrained* oracle, whereas the
  shipped results had the *long-only* value there;
* `turnover` is `None` (undefined) when no weight history is supplied, instead
  of a misleading `0.0`.

An earlier draft of this audit also reported a spurious ρ = −0.939 between two
derived columns; that was a tie-handling bug on a constant column in the
analysis code, not in the evaluators. The honest value is −0.129.

---

## 5. What the bundle contains now

`04_evaluation_results/complete_evaluation.json` **has been recomputed** for the
`pnl` and `portfolio` blocks with the fixed evaluators
(`scripts/handover/recompute_downstream.py`), from the same stored arrays the
original run used — the generated tensors in `02_generated_outputs/` and the
verified references in `03_ground_truth/reference/`. Each cell carries a
`downstream_recomputed_with_fixed_evaluators` record with the sample count, the
reference pool, whether the shipped `GMV_LongOnly` was still 1/N, and the
shipped values of the three affected headline metrics.

Unaffected by the fixes, and therefore **byte-identical to the original run**:
`FidelityEvaluator`, `DiversityEvaluator`, `StylizedFactsEvaluator` and
`utility` (deep hedging). The audit asserts this per cell.

The **original as-shipped run** is preserved unmodified at
`outputs/results/latest/evaluation/complete_evaluation.json` — compare against it
to see exactly what each fix moved. Worst relative change in
`terminal_wealth_mean` across all 96 arms: **1.64 %**.

| column | before | after |
|---|---|---|
| `pnl.*.sortino_ratio` | up to 6.65e12 | max 58.46, undefined paths counted |
| `pnl.*.terminal_wealth_*`, `max_drawdown` | `prod(1+r)` on log returns | `exp(sum r)`, ≤1.64 % apart |
| `portfolio.*.GMV_LongOnly.*` | == `1N` in 144/144 | a genuine optimum in 144/144 |
| `portfolio.*.Real_GT_*` | 2 colliding keys, long-only values | 4 distinct keys |
| `portfolio.*.turnover` | 0.0 in 1008/1008 | `null` (no weight history) |

Still **not** fixed (they need the models, torch and the compute node):

1. **R1/D3 sample sizing** — the real reference is still the DL set's test split,
   truncated per model. At L=252 that is 5–9 windows. Treat any L=252 column as a
   sanity check, not a comparison.
2. **B3/B4** — overlapping real windows and a model-dependent real baseline.
   `pnl.real.*` is not the same series across families; compare within a family,
   or use within-cell gaps.
3. **P4** — no rolling estimation, no transaction costs.
4. **B5** — `equal_weight_bh` remains model-free.

## 6. Data caveats in the results JSON

* The file is **Python-extended JSON**: it holds 20 `NaN` literals, all in
  `StylizedFactsEvaluator.long_memory_volatility.{real,diff}` for 10 of the 48
  cells. They are inherited unchanged from the shipped run. Pass
  `parse_constant=` to a strict parser if you need standard JSON.
* `portfolio.*.turnover` is `null` rather than `0`, and `pnl.*.sortino_ratio`
  can be `null`/`inf` where a path has no downside — the companion
  `n_sortino_defined` / `n_sortino_undefined` / `n_omega_infinite` / `n_paths`
  fields say how many paths the mean was taken over.
* Note that `complete_evaluation.json` was not regenerated end-to-end; only the
  `pnl` and `portfolio` blocks were recomputed (`recompute_downstream.py`). The
  rest is the shipped run verbatim, asserted per cell by `verify_bundle.py` and
  `audit_downstream.py`.

## 7. What a correct evaluation would need

1. **A single, model-independent real reference** — one fixed out-of-sample
   block per length, used by every model, with the generated side truncated
   rather than the reference side (or both subsampled to a common, documented n).
2. **Non-overlapping (or block-bootstrapped) evaluation paths** for the real
   arm, so per-path statistics are not 99 % duplicated.
3. **A real pool sized by the statistical set** (596 rows) rather than the DL
   set's 260-row test split, so L=252 is not starved.
4. **Rolling out-of-sample portfolio estimation** using `estimation_window` /
   `rebalance_freq`, with turnover and transaction costs actually recorded.
5. **Ratio metrics reported with their path counts**, undefined ratios dropped
   rather than epsilon-patched.
6. **A model-free P&L strategy** (or an explicit statement that
   `equal_weight_bh` measures data properties, not decision quality).
