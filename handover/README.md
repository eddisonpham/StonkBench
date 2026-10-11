# Handover — StonkBench evaluation of 12 generative market models

Data-only bundle: **checkpoints, generated samples, the real data they were
scored against, and the evaluation results for all 48 cells**. No figures.

Start with [`GUIDE.md`](GUIDE.md) — copy-paste recipes for comparisons, sequence-
length ablations, visualisations and new metrics, plus a task table for coding
agents. Then read [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md): it lists the defects that
were found in the P&L and portfolio evaluators, what was corrected in the results
JSON, and what is still unfixed.

## Layout

| path | contents |
|---|---|
| `01_model_checkpoints/` | **the `latest/` checkpoint set — the one the results reference** (7 DL models, 10 files), `source_manifests/latest_MANIFEST.json`, `checkpoint_inventory.csv` |
| `02_generated_outputs/` | the 48 generated sample tensors, one per model × length, `(n_samples, L, 25)`; `generated_output_inventory.csv` |
| `03_ground_truth/` | `ground_truth_seq{L}.pt` (canonical window sets), `eval_reference/` (the set the run's output directory held), and `reference/<model>_seq<L>.npz` — **the verified real data each cell was actually scored against** |
| `04_evaluation_results/` | `complete_evaluation.json` (all 48 cells, all taxonomies, corrected downstream) and `evaluation_metadata.json` (alignment, metric directions, fix list) |
| `MANIFEST.json` | every file: path, bytes, sha256, kind, model, sequence length |

## Scale

12 models (5 statistical, 7 deep learning) × 4 lengths {21, 42, 126, 252} = **48
cells**, 25 assets. Taxonomies: fidelity, diversity, stylised facts, P&L, plus
portfolio construction and deep-hedging utility.

## Using it

```python
import sys; sys.path.insert(0, "scripts/handover")
from load_handover import Handover

h = Handover()                              # finds handover/ automatically
h.results()["quantgan_seq126"]              # the full evaluation dict for one cell
h.generated_outputs("quantgan", 126)        # (9, 126, 25) float32
h.reference("quantgan", 126)                # the real data it was scored on
h.effective_n("quantgan", 126)              # 9
h.metric("quantgan_seq126", "pnl.synthetic.sharpe_ratio")
h.metadata()["alignment"]                   # per-cell sample counts + proof deltas
h.checkpoint("quantgan")                   # the checkpoint the results used -> numpy arrays
h.checkpoint_inventory()                    # model, file, sha256, referenced_by_results
```

Only `numpy` is needed to read the bundle (`scripts/handover/torch_pt_reader.py`
parses `.pt` files directly); `scipy` is needed to re-run the portfolio
optimisers. Torch is only needed to *use* the checkpoints.

```bash
python3 scripts/handover/build_manifest.py          # rebuild MANIFEST.json
python3 scripts/handover/verify_bundle.py           # integrity + coverage + consistency
PYTHONPATH=<libs> python3 scripts/handover/audit_downstream.py    # defect evidence
PYTHONPATH=<libs> python3 scripts/handover/reconstruct_references.py --no-write
PYTHONPATH=<libs> python3 scripts/handover/recompute_downstream.py   # rerun the corrected blocks
```

## Read this before comparing anything

1. **Sample counts are not equal.** Five DL models (`cond_sig_wgan`,
   `conditional_tsdiffusion`, `quantgan`, `timegrad`,
   `unconditional_tsdiffusion`) ship only **9** generated samples; the other seven
   ship 1000. The evaluator aligns both sides with `n = min(reference, samples)`,
   so the *effective* n is 240/219/135/9 (statistical), 120/110/68/5
   (`kalman_vae_safe`, `vrnn_klsched`) and 9/9/9/5 (the nine-sample models). See
   `metadata()["alignment"]`.
2. **L=252 is unusable for model comparison** — every cell was scored on at most
   9 real windows, DL cells on 5 (see `KNOWN_ISSUES.md` §3).
3. **The real reference is not fixed across models.** Statistical cells are
   scored against the whole stride-1 window pool, DL cells against its second
   half, each truncated to the aligned n. So `pnl.real.*` is not the same series
   across families. Compare within a family, or use within-cell gaps.
4. **DL models were trained once at L=252.** Shorter-length cells are the same
   checkpoint trimmed, not four independent trainings — an "ablation over L" is
   an ablation over *evaluation window*, not training length.
5. **Statistical models have no trainable checkpoint.** They are fitted per
   generation; configuration is in the artifact metadata
   (`02_generated_outputs/<model>/<model>_seq<L>.pt` → `metadata`). The 10
   shipped checkpoints are all W=252-trained; `referenced_by_results` in
   `checkpoint_inventory.csv` says which 6 of them the results' own
   `model_checkpoint_manifest` names (the others are legacy-named companions
   `fit()` keeps, plus `cond_sig_wgan_G.pt`).
6. **No downstream checkpoints exist.** The hedging/portfolio models are trained
   in-process during evaluation and were not persisted; only their results are
   here.
7. **`pnl` and `portfolio` in the results JSON are corrected**, not as-shipped.
   The original run is at
   `outputs/results/latest/evaluation/complete_evaluation.json`; fidelity,
   diversity, stylised facts and hedging are byte-identical between the two.
8. **n = 12 models**, several of them singletons in their family. Cross-model
   claims are directional only.

## Provenance

* built from the StonkBench working tree at `outputs/results/latest/` and
  `outputs/ground_truth/`; every file's sha256 is in `MANIFEST.json`
* only the checkpoint set the results used is shipped (`latest/`); earlier
  per-length and alternate runs were removed, and `checkpoint_inventory.csv`
  records a sha256 for each remaining file
* the per-cell references in `03_ground_truth/reference/` are **verified**: they
  reproduce the metric values recorded for every cell to ≤ 5e-9
  (`scripts/handover/reconstruct_references.py`)
* tooling lives in `scripts/handover/`, not in the bundle
