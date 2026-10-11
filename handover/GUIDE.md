# Guide — what to do with this bundle

A cookbook for a human **or a coding agent** that wants to produce
visualisations, comparisons or ablations from the 48 evaluated cells. Every
snippet is copy-paste runnable against this bundle with `numpy` only.

```bash
cd <repo root>
export PYTHONPATH=<site-packages-with-numpy>:scripts/handover
python3 - <<'PY'
from load_handover import Handover
h = Handover()
print(len(h.results()), "cells")
PY
```

`torch` is **not** needed to read the bundle: `scripts/handover/torch_pt_reader.py`
parses `.pt` files directly. `scipy` is only needed to re-run the portfolio
optimisers. See [`README.md`](README.md) for the layout and
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) before you quote any number.

---

## 0. The one thing to read first

Two of the six metric blocks in `complete_evaluation.json` are **corrected**
(`pnl`, `portfolio`) and four are **the shipped run verbatim** (`FidelityEvaluator`,
`DiversityEvaluator`, `StylizedFactsEvaluator`, `utility`). Per cell,
`downstream_recomputed_with_fixed_evaluators` records what was recomputed, on how
many samples, and the values the shipped run had. The `L=252` column is a sanity
check only (5–9 real windows). All of this is machine-readable in
`04_evaluation_results/evaluation_metadata.json`.

---

## 1. Load anything in three lines

```python
from load_handover import Handover
h = Handover()                                  # auto-finds ./handover

X = h.generated_outputs("quantgan", 126)        # (9, 126, 25) float32 synthetic
R = h.reference("quantgan", 126)                # (9, 126, 25) real windows it scored against
n = h.effective_n("quantgan", 126)              # 9  == X.shape[0] == R.shape[0]
cell = h.results_for("quantgan", 126)           # the whole evaluation dict
val = h.metric("quantgan_seq126", "pnl.synthetic.sharpe_ratio")
meta = h.metadata()                             # alignment, metric_directions, fixes
ckpt = h.checkpoint("quantgan")                 # the checkpoint the results used
```

Reference of the API: `Handover.cells()`, `.generated_outputs()`, `.ground_truth(L)`,
`.reference()`, `.effective_n()`, `.alignment()`, `.cell_metadata()`,
`.checkpoint(model, which="referenced"|"any")`, `.checkpoint_inventory()`,
`.results_for()`, `.metric()`, `.files(kind)`. Every accessor raises `KeyError`
listing the valid keys.

Raw access without the API (useful in a one-liner):

```python
import numpy as np
from torch_pt_reader import load_pt
X = load_pt("handover/02_generated_outputs/quantgan/quantgan_seq126.pt")["data"]
R = np.load("handover/03_ground_truth/reference/quantgan_seq126.npz")["data"]
```

---

## 2. Where every number lives

```
cell = h.results_for(model, L)
cell["FidelityEvaluator"]   .kd .md .mdd .sd .sdd .cmd .dcor_diff      # lower is better
cell["FidelityEvaluator_per_channel"]                                  # 25 rows
cell["DiversityEvaluator"]  .icd_euclidean .icd_dtw                    # no target; see §4
cell["StylizedFactsEvaluator"].{autocorr_returns,volatility_clustering,
                               long_memory_volatility}.{real,synth,diff}
cell["utility"].summary.augmented_testing.{BlackScholes,DeltaGamma,Feedforward_L-1,
                               Feedforward_Time,LinearRegression,LSTM,RNN,XGBoost}
cell["portfolio"].n_assets_{5,10,25}.{1N,GMV,GMV_LongOnly,Tangency,Tangency_LongOnly,
                               Real_GT_RealGMV,Real_GT_RealGMV_LongOnly,
                               Real_GT_RealTangency,Real_GT_RealTangency_LongOnly}
                 .{sharpe_ratio,sortino_ratio,max_drawdown,calmar_ratio,omega_ratio,
                   tail_ratio,annualized_return,annualized_volatility,turnover,
                   herfindahl,ceq_gamma_1.0,ceq_gamma_2.0,ceq_gamma_5.0,weights}
cell["pnl"].{real,synthetic}.{13 metrics + n_paths,n_sortino_defined,
                             n_sortino_undefined,n_omega_infinite}
cell["pnl"].terminal_wealth_wasserstein .strategy .strategy_weights
```

Use `h.metadata()["metric_directions"]` for `direction` (lower/higher) and `status`
(`ok` / `broken` / `degenerate` / `suspect`) per metric path. **Never rank a
metric whose status is not `ok`** — that is the whole point of the column.

---

## 3. Recipe: cross-model comparison at one length

```python
import numpy as np
from load_handover import Handover
h = Handover()

L = 126
rows = []
for c in h.cells():
    if c.sequence_length != L:
        continue
    cell = h.results_for(c.model, L)
    f = cell["FidelityEvaluator"]
    rows.append((c.model, c.model_type, h.effective_n(c.model, L),
                 f["kd"], f["mdd"], f["cmd"], f["dcor_diff"],
                 cell["pnl"]["terminal_wealth_wasserstein"],
                 cell["StylizedFactsEvaluator"]["volatility_clustering"]["diff"]))

hdr = ("model", "type", "n", "kd", "mdd_w1", "cmd", "dcor_diff", "tw_wass", "volclust_diff")
print(("{:<26}" + "{:>12}" * 8).format(*hdr))
for r in sorted(rows, key=lambda r: r[3]):
    print("{:<26}{:>12}{:>12}{:>12.4f}{:>12.5f}{:>12.4f}{:>12.4f}{:>12.5f}{:>12.4f}".format(*r))
```

Sort by the metric you care about; print `n` alongside it — sample counts differ
by up to 27x across cells (see `alignment()`), and a model with `n=9` is not
comparable to one with `n=240`.

## 4. Recipe: ablation over sequence length

```python
import numpy as np
from load_handover import Handover
h = Handover()

models = sorted({c.model for c in h.cells()})
metric = lambda cell: cell["FidelityEvaluator"]["kd"]      # any dotted path
print(f"{'model':<26}" + "".join(f"{L:>12}" for L in (21, 42, 126, 252)))
for m in models:
    vals = [metric(h.results_for(m, L)) for L in (21, 42, 126, 252)]
    print(f"{m:<26}" + "".join(f"{v:>12.4f}" for v in vals))
```

Read it as an ablation over **evaluation window**, not training length: the DL
models were trained once at W=252 and trimmed, and the L=252 column is estimated
on ≤9 real windows. For a paired view, use the within-cell gaps
(`pnl.synthetic.X - pnl.real.X`) rather than the raw values.

## 5. Recipe: synthetic vs reference plot

```python
import numpy as np, matplotlib.pyplot as plt
from load_handover import Handover
h = Handover()

model, L, ch = "quantgan", 126, 0
X, R = h.generated_outputs(model, L), h.reference(model, L)
fig, ax = plt.subplots(figsize=(11, 4))
for i in range(min(20, R.shape[0])):
    ax.plot(R[i, :, ch], color="black", alpha=.35, lw=.9)
for i in range(min(20, X.shape[0])):
    ax.plot(X[i, :, ch], color="tab:orange", alpha=.35, lw=.9)
ax.plot([], [], color="black", label=f"real (n={R.shape[0]})")
ax.plot([], [], color="tab:orange", label=f"synthetic (n={X.shape[0]})")
ax.set_title(f"{model} — seq {L} — channel {ch}")
ax.legend(); fig.tight_layout(); fig.savefig(f"{model}_seq{L}_ch{ch}.png", dpi=150)
```

Swap `ch` for `h.cell_metadata(model, L)["asset_columns"][ch]` for the ticker, and
use `X[:, :, :5].mean(axis=2)` for an equal-weight portfolio path.

## 6. Recipe: downstream comparison (P&L / portfolio)

```python
from load_handover import Handover
h = Handover()

for c in h.cells():
    p = h.results_for(c.model, c.sequence_length)["pnl"]
    print(f"{c.key:<28} sharpe gap {p['synthetic']['sharpe_ratio'] - p['real']['sharpe_ratio']:+7.3f}"
          f"  tw_wass {p['terminal_wealth_wasserstein']:.5f}"
          f"  n={p['synthetic']['n_paths']}"
          f"  sortino_undef={p['synthetic']['n_sortino_undefined']}")
```

Portfolio: compare `portfolio.n_assets_25.GMV_LongOnly.sharpe_ratio` against the
two brackets that bound it — `1N` (naive) and `Real_GT_RealTangency` (oracle). A
number without both brackets is uninterpretable. `turnover` is `null` (no
rebalancing history exists), and `weights` is the exact vector used.

## 7. Recipe: compute a new metric

`X` and `R` are plain `(n, L, 25)` arrays of **log returns**, so anything in
`src/taxonomies/` works on them directly:

```python
import sys; sys.path.insert(0, ".")
from torch_stub import install_torch_stub
install_torch_stub()                          # lets src.taxonomies import without torch
from src.taxonomies.pnl import _pnl_metrics
from load_handover import Handover
h = Handover()

X = h.generated_outputs("quantgan", 126)
print(_pnl_metrics(X.mean(axis=2)))          # equal-weight portfolio over the 25 channels
```

If you write a *new* evaluator, follow the rules in
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) §7: declare the direction (or make it
descriptive), say whether it is a score / reference / treatment arm, compute it at
a fixed sample size across models, and report it per asset before averaging.

---

## 8. If you are a coding agent

**Goal templates** — each is a bounded task; all of them are answered by the
snippets above.

| asked to… | start from | mind |
|---|---|---|
| plot / visualise | §5 | no figures ship in the bundle; generate your own |
| compare models | §3 | print `n` next to every metric |
| ablate over sequence length | §4 | L=252 is not comparable; it is an evaluation-window ablation |
| rank models | §2 + `metric_directions` | drop every `status != ok` metric first |
| reproduce a number | `h.results_for(...)` | the JSON *is* the record; nothing needs re-running |
| fix a downstream evaluator | `src/taxonomies/{pnl,portfolio}.py` | then `recompute_downstream.py`, then `verify_bundle.py` |
| regenerate the bundle | `scripts/handover/` | `build_manifest.py` + `verify_bundle.py` must both pass |

**Invariants** (all asserted by `scripts/handover/verify_bundle.py`, exit 0 = safe
to proceed): 48 cells; 48 generated tensors; 48 verified references whose row
count equals `n_used_by_evaluator`; 10 checkpoints, all from the `latest` set;
9 portfolio strategies per cell with 4 distinct `Real_GT_*` keys; no `NaN`-inflated
ratios; no image files; every file hashed in `MANIFEST.json`.

**Do not**: re-derive metrics from the generated tensors and mix them into the
results JSON (they will not match the recorded ones); treat `pnl.real.*` as one
fixed reference across models; average the two arms of the hedging experiment;
or rank raw `icd_*` diversity values.

**Reproduce the evidence**:

```bash
python3 scripts/handover/verify_bundle.py            # integrity + coverage + consistency
python3 scripts/handover/audit_downstream.py         # defect evidence + post-fix checks
python3 scripts/handover/reconstruct_references.py --no-write   # reference proof (4.9e-09)
python3 scripts/handover/recompute_downstream.py     # idempotent rewrite of pnl+portfolio
```
