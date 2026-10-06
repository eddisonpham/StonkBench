# Full-scale ablation evaluation plan — EVALS STAGED, NOT RUN

Status: **generation complete and verified (all model families); evals staged
only.** Per the operator decision (2026-10-05) the taxonomy evals and
downstream tasks are *prepared* here but must not be executed until
explicitly requested.

Generation: jobs 1038126 (84 DL array tasks, 84/84 rc=0) + 1038516 (cancelled
transient, resubmitted) + 1038569 (cond_sig_wgan provenance fix) + 1038717 /
1038813 (60 statistical cells). Verified by `verify_generation.py`:
**ALL CHECKS PASSED on 456 checks — 432 artifacts (3 ablations × 12 models ×
4 seqs × 3 batches) + 24 ground-truth files** (shape, finiteness,
uniqueness, metadata contract, pairwise-distinct batches, batch-1 std within
0.986–1.020 of the shipped `simulation_v2` artifacts for the 7 DL models and
0.980–1.065 of the production `latest` artifacts for the 5 statistical
baselines).

Machine-readable index: `outputs/generation/EVAL_INPUTS.{json,csv}`
(432/432 cells, zero missing, with per-cell paths/seeds/checkpoints and the
ready-to-run eval commands).

## 1. Inputs

### Checkpoints (in-repo, durable)

```
checkpoints/simulation_v2/<model>/<model>_seq{21,42,126,252}_final.pt   # 28 files
checkpoints/fair_20261004/<model>/..._seq252_final.pt                    # seq-252 only
checkpoints/latest/<model>/..._seq252_final.pt                           # seq-252 only
checkpoints/<ablation>/MANIFEST.json                                     # sha256 + source path + seed
```

Every DL adapter now implements the evaluator's regeneration contract
(`can_regenerate_from_checkpoint = True`, `load_state(checkpoints)`):

| model | checkpoint shape | arch recovery |
|---|---|---|
| `vrnn_klsched` | bare state_dict | x/h/z dim + GRU layers from tensor shapes |
| `kalman_vae_safe` | bare state_dict | a_dim/z_dim/K from `_mat_*`; image_side from encoder fan-in; channels from dl_set |
| `cond_sig_wgan` | bare generator state_dict | hidden dims + `p` from linear shapes; `noise_std` buffer re-registered |
| `quantgan` | `{config_snapshot, per_asset_state_dicts…}` | rebuild `QuantGANTrainer` per channel |
| `timegrad` | `{lags/context/cells/diff…, state_dict}` | rebuild `TimeGradTrainingNetwork` (embedding_dim=1, as in `fit`) |
| `unconditional_tsdiffusion` | `{context/pred/input_dim, state_dict}` | rebuild `TSDiff` from `diffusion_small_config` |
| `conditional_tsdiffusion` | same | inherited; `_build_model` swaps in `TSDiffCond` |

Verified: `scripts/mini_exp/smoke_load_state.py` passes all 28 (model, L) cells;
at seq42 the 500-sample regeneration matches the shipped `simulation_v2`
artifacts distributionally (std ratio 0.984–1.012, channel ratios 0.95–1.05).

### Generated sequences

```
<out_root>/<ablation>/batch{1,2,3}/<model>/artifacts/<model>_seq{L}.pt
<out_root>/<ablation>/batch{1,2,3}/<model>/gen_seq{L}.json        # sidecar per cell
<out_root>/<ablation>/ground_truth/ground_truth_seq{L}.pt         # dl_set-derived (DL)
<out_root>/<ablation>/ground_truth/statistical/ground_truth_seq{L}.pt  # stats_set test span
<out_root>/EVAL_INPUTS.{json,csv}                                 # index of every cell
```

Twelve models per ablation: the 7 DL checkpoint models **plus the 5
statistical baselines** (`block_bootstrap`, `stationary_block_bootstrap`,
`merton_jump_diffusion`, `de_jump_diffusion`, `garch11`) — those have no
checkpoints, so the job re-fits them on the `stats_set` train split
(sub-second) and samples natively. Their artifacts stay in **raw space** with
`model_type=statistical` / `fit_from_stats_set=true`, matching
`run_model_experiment`, and the evaluator routes them to the stats_set
real-data reference. Because the statistical fits and batch seeds do not
depend on the ablation, those cells are byte-identical across the three
ablation trees (correct: the baseline is ablation-independent).

`<out_root>` = `${OUTPUT_ROOT}/generation` = `/scratch/epham/stonkbench/output/generation/`
for the Slurm run (compute nodes cannot write `/home`), already **rsynced to
`<repo>/outputs/generation/`** (2.3 G — 432 artifacts + 24 ground-truth files +
432 sidecars + the EVAL_INPUTS index). The repo copy is the canonical one;
re-sync after any re-run:

```bash
rsync -a /scratch/epham/stonkbench/output/generation/ \
         /home/epham/StonkBench/outputs/generation/
```

Semantics (identical to `src/experiments/core/pipeline.py`):

* 3 ablations × 7 models × 4 seq lengths × 3 batches × 500 samples.
* native draw length = the checkpoint's trained window (`L` for `simulation_v2`,
  252 for `fair_20261004`/`latest`), trimmed with `stitch_sequences` (prefix
  slice) — recorded as `native_length` / `stitched_from_native` in metadata.
* samples are denormalized with `channel_norm_stats(dl_set_W{native})`.
* batch seeds `1042 / 2042 / 3042` (`seed_base 42 + 1000·batch`), distinct from
  the original run's 42 and the probe seeds 43/44/45.

## 2. Taxonomy evals (staged — do not submit yet)

`scripts/mini_exp/eval_gen_batches.sh` — array `0-8`, one task per
(ablation, batch); each task owns its `results_dir`
(`…/<ablation>/evaluation_batch<k>/`), so the 9 writers never race on
`complete_evaluation.json`. Each tree holds **12 models × 4 seq lengths = 48
artifacts**, i.e. 432 evaluations in total (the simulation_v2 suite of 28
artifacts took 9:29 h serially, so budget accordingly or shard with
`EVAL_MODEL` / `EVAL_SEQ`).

```bash
# from trig-login01, when green-lit:
command sbatch --export=ALL scripts/mini_exp/eval_gen_batches.sh

# single cell (mirrors eval_cell.sh):
command sbatch --export=ALL,EVAL_ABLATION=simulation_v2,EVAL_BATCH=1,\
EVAL_MODEL=timegrad,EVAL_SEQ=126 --array=0 scripts/mini_exp/eval_gen_batches.sh
```

Each task runs the standard suite (fidelity, diversity, stylized facts,
visual) **and** the downstream metrics the evaluator already owns:

* `src/taxonomies/portfolio.py` — portfolio optimisation (DeMiguel et al.)
* `src/taxonomies/pnl.py` + hedging error — P&L / hedging downstream
* utility (`UtilityMetricsEvaluator`, legacy mse hedger path)

Real-data side stays pinned to `data/preprocessed/dl_set.pt` (W252), exactly the
protocol that produced the accepted `PASS=19 / WARN=9` scorecard, so the new
numbers remain comparable with `MODEL_PAPER_AUDIT.md` §10.

**Do not pass `--seq_lengths`**: artifacts are already stored at their target
length; the filter would trim the *last* L steps instead of scoring natively.

## 3. Downstream tasks (staged)

`spec.md` asks for "as many synthetic samples as there are in the test set for
downstream evaluation". Each artifact carries `num_samples` (500) and the
batch/seed in metadata, so a downstream runner can subsample deterministically
to the test-window count (`len(test_series) - L + 1`) per ablation without
regenerating.

Planned order once generation is verified:

1. Run `eval_gen_batches.sh` (9 trees) → `evaluation_batch{1,2,3}/complete_evaluation.json`.
2. Aggregate the 3 batches per (model, L, ablation) with the staged
   `scripts/mini_exp/aggregate_batch_metrics.py` (already fixture-tested): mean
   ± std of every numeric metric → `batch_metrics_aggregate.{json,csv}` — the
   reason 3 replicate batches exist.
3. Score batch-to-batch variance against the ±tolerance bands that produced the
   §10 WARN list; investigate any cell whose batch spread crosses a band.
4. Feed the aggregate into `MODEL_PAPER_AUDIT.md` as the full-scale ablation
   table (v2 vs fair_20261004 vs latest).

## 4. Generation jobs

```bash
# from trig-login01:
command sbatch --export=ALL scripts/mini_exp/gen_batches.sh         # array 0-83  (7 DL models)
command sbatch --export=ALL scripts/mini_exp/gen_batches_stat.sh    # array 0-59  (5 statistical models)
```

Array map (DL): `tid/28 -> ablation`, `(tid%28)/4 -> model`, `tid%4 -> seq`.
Array map (statistical): `tid/20 -> ablation`, `(tid%20)/4 -> model`, `tid%4 -> seq`.

Re-running is idempotent: existing artifacts are skipped
(`STONKBENCH_GEN_FORCE=1` overrides), and each artifact is published atomically
(`.tmp<pid>.pt` → rename). After any re-run, refresh the index:

```bash
python scripts/mini_exp/build_eval_manifest.py --root <out_root>
```
