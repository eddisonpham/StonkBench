# Full-scale ablation evaluation plan — EVALS RUNNING (submitted 2026-10-06)

Status: **generation complete and verified (all model families); taxonomy +
downstream evals SUBMITTED and running as one job per artifact.**

* smoke: job **1040042** (debug partition, 1 cell — `block_bootstrap` @ seq21,
  `simulation_v2`/batch1) → rc=0 in 6:13, all 22 metric blocks present, plots
  written.
* full run: job **1040043**, array **0-431** = one artifact per task (3
  ablations × 3 batches × 12 models × 4 seq lengths), submitted from
  `trig-login01` with
  `command sbatch --export=ALL,GEN_ROOT=/scratch/epham/stonkbench/output/generation scripts/mini_exp/eval_gen_batches.sh`.
* progress: `bash scripts/mini_exp/await_eval_jobs.sh 1040158,1044081,1040509`
  (report only), log `/tmp/await_eval.log`. The monitor warns about historical
  FAILED/TIMEOUT states without exiting — `sacct` state is sticky, so a task
  that was already re-queued elsewhere would otherwise kill it on poll 1; the
  authoritative gate is `check_eval_coverage.py`.
* short-walltime re-submission: task 11-431 of 1040043 was cancelled once the
  measured cost per cell was known and re-submitted as 
  **1040158** (`--array=24-431`, `--time=02:00:00`) so the tasks backfill into
  smaller gaps. The script skips cells that already have a valid summary, so a
  re-submission never recomputes finished work (`EVAL_FORCE=1` overrides).
* top-ups: task 26 timed out at the 40-min limit → re-queued as **1040509**
  (`merton_jump_diffusion` @ seq126). Four tasks (153/154/157/159) were
  cancelled after starting (cells `cond_sig_wgan`@42/126 and
  `de_jump_diffusion`@42/252 in `fair_20261004`/batch1) → resubmitted as
  **1044081** (`--array=153,154,157,159`, 02:00:00). The skip guard makes every
  re-submission idempotent.
* finisher: **1044082**, chained with
  `--dependency=afterany:1040158:1040509:1044081` (1040043 already finished,
  so its leg was satisfied at submit). It runs `check_eval_coverage.py` and,
  only when all 432 cells are present and well-formed,
  `aggregate_batch_metrics.py`. It never resubmits work: if the grid is
  incomplete it prints the missing cells and exits 1.
* after all 432 tasks: `aggregate_batch_metrics.py` (see §3) — there is no
  merge step any more.

### Measured cost per cell (why this takes hours)

Wall time per cell is dominated by the downstream block, whose cost scales as
`(T - L + 1) * L` — the number of stride-1 windows of length `L` in the test
span `T` times the length of each window (the hedger is re-fit per window).
That product peaks at `L = T/2 = 126`, which is exactly what the scheduler
reports:

| seq length | typical task elapsed |
|---|---|
| 21 | 4-7 min |
| 42 | 7-17 min |
| **126** | **16-52 min** |
| 252 | 10-19 min |

So the six `seq126` shards were not only the ones hit by the missing thread
clamp (now fixed); `seq126` is intrinsically the most expensive cell, and the
108 `seq126` cells dominate the remaining compute (~60 of ~115 CPU-hours).
This is real downstream training, not a bug: the number of windows and the
per-window fit both enter.

Wall-clock to finish is queue-bound, not code-bound: the site's submit wrapper
*requires* a GPU request (`--gpus-per-node`), so no CPU-only submission can
sidestep the GPU queue, and the account holds only the `normal` QOS (no
`bigprio`/`bump`), so scheduling priority is whatever fairshare grants. At ~5
concurrent tasks (4-15 observed) the remaining grid is on the order of 12-20 h;
the finisher above still closes it out unattended.

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
`merton_jump_diffusion`, `de_jump_diffusion`, `dcc_garch`) — those have no
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

## 2. Taxonomy + downstream evals (submitted — job 1040043, array 0-431)

`scripts/mini_exp/eval_gen_batches.sh` — array **0-431**, **one artifact per
task**:

| `tid` expression | mapping |
|---|---|
| `tid / 144` | ablation (`simulation_v2`, `fair_20261004`, `latest`) |
| `(tid / 48) % 3 + 1` | batch 1..3 |
| `(tid / 4) % 12` | model (`MODELS[]` in the script) |
| `tid % 4` | seq 21 / 42 / 126 / 252 |

`tid / 144 → ablation` and the model order are asserted against the tree:
the 432 `(ablation, batch, model, seq)` tuples the formula generates are an
exact match for the 432 artifacts on disk (checked before submitting).

Each task runs its cell end to end — fidelity, diversity, stylized facts,
visual + the downstream metrics the evaluator owns (`utility` + hedging,
`portfolio` optimisation, `pnl`) — and then that cell's plots. It owns
`results_dir = <ablation>/evaluation_batch<k>/<model>_seq<L>/`, because the
evaluator **overwrites** `complete_evaluation.json` on every run: one cell per
directory means 432 concurrent writers never race and no merge step is needed.

Walltime 06:00:00 per task: measured ~6 min for a statistical cell and
single-digit minutes for DL cells once the thread clamp below is in place
(the earlier ~18 min/artifact came from unclamped OpenMP pools).

Two fixes landed with this restart:

* **Thread clamp (this was the real slowness).** The script never exported
`OMP_NUM_THREADS` (the generation drivers did), so OpenMP sized its pool from
the node's ~96 cores instead of the 8 allocated CPUs and spin-waited inside the
affinity mask, starving the working thread — 7–9 CPU-hours per artifact. It now
exports `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`NUMEXPR_NUM_THREADS` = 8 and `OMP_WAIT_POLICY=passive`, and logs them at
startup (`slots=8 OMP_NUM_THREADS=8 …`).
* **Exact model filter.** `--model` used to match as a substring, so
  `block_bootstrap` also matched `stationary_block_bootstrap`; the evaluator now
  matches exactly and each task verifies its summary holds exactly its own
  cell.

```bash
# from trig-login01 (already submitted as job 1040043):
command sbatch --export=ALL,GEN_ROOT=/scratch/epham/stonkbench/output/generation \
  scripts/mini_exp/eval_gen_batches.sh

# preview what an index maps to (no work done):
EVAL_DRY_RUN=1 SLURM_ARRAY_TASK_ID=431 bash scripts/mini_exp/eval_gen_batches.sh

# single cell:
command sbatch --export=ALL,EVAL_ABLATION=simulation_v2,EVAL_BATCH=1,\
EVAL_MODEL=timegrad,EVAL_SEQ=126 --array=0 scripts/mini_exp/eval_gen_batches.sh
```

Per-cell results layout:

```
<root>/<ablation>/evaluation_batch<k>/<model>_seq<L>/
    complete_evaluation.json          # exactly one entry, keyed <model>_seq<L>
    seq_<L>/<model>/{metrics.json,per_asset,visualizations,…}
    plots/{comparison,downstream,per_asset,scatter,window_size}
```

Each task runs the standard suite (fidelity, diversity, stylized facts,
visual) **and** the downstream metrics the evaluator already owns:

* `src/taxonomies/portfolio.py` — portfolio optimisation (DeMiguel et al.)
* `src/taxonomies/pnl.py` + hedging error — P&L / hedging downstream
* utility (`UtilityMetricsEvaluator`, legacy mse hedger path)

Real-data side stays pinned to `data/preprocessed/dl_set.pt` (W252), exactly the
protocol that produced the accepted `PASS=19 / WARN=9` scorecard, so the new
numbers remain comparable with `MODEL_PAPER_AUDIT.md` §10.

**Do not pass `--seq_lengths`** (plural): artifacts are already stored at their
target length; that filter would trim the *last* L steps instead of scoring
natively. The script passes `--seq_length` (singular), which keeps only the
artifact whose native length equals L.

## 3. Downstream tasks (staged)

`spec.md` asks for "as many synthetic samples as there are in the test set for
downstream evaluation". Each artifact carries `num_samples` (500) and the
batch/seed in metadata, so a downstream runner can subsample deterministically
to the test-window count (`len(test_series) - L + 1`) per ablation without
regenerating.

Planned order once generation is verified:

1. Run `eval_gen_batches.sh` (array 0-431 = one artifact per task) →
   `evaluation_batch{k}/<model>_seq{L}/complete_evaluation.json` (1 cell each).
2. (No merge step: the per-cell summaries *are* the record; they are read
   directly by the aggregator.)
3. Aggregate the 3 batches per (model, L, ablation) with
   `scripts/mini_exp/aggregate_batch_metrics.py` (already fixture-tested): mean
   ± std of every numeric metric → `batch_metrics_aggregate.{json,csv}` — the
   reason 3 replicate batches exist.
4. Score batch-to-batch variance against the ±tolerance bands that produced the
   §10 WARN list; investigate any cell whose batch spread crosses a band.
5. Feed the aggregate into `MODEL_PAPER_AUDIT.md` as the full-scale ablation
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
