#!/usr/bin/env python3
"""Verify the handover bundle: integrity, coverage, consistency, no figures.

    python3 scripts/handover/verify_bundle.py

Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_handover import MODEL_TYPES, SEQUENCE_LENGTHS, Handover  # noqa: E402

H = Handover()
ROOT = H.root
FAILURES: list[str] = []
FIGURE_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps"}


def check(cond: bool, msg: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAILURES.append(msg)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_manifest() -> None:
    print("\n[1] manifest integrity")
    files = H.manifest()["files"]
    check(len(files) > 0, f"manifest lists {len(files)} files")
    missing, wrong_size, wrong_hash = [], [], []
    for rec in files:
        p = ROOT / rec["path"]
        if not p.exists():
            missing.append(rec["path"])
            continue
        if p.stat().st_size != rec["bytes"]:
            wrong_size.append(rec["path"])
            continue
        if sha256_of(p) != rec["sha256"]:
            wrong_hash.append(rec["path"])
    check(not missing, f"all {len(files)} files exist ({len(missing)} missing)")
    check(not wrong_size, f"all recorded sizes match ({len(wrong_size)} mismatched)")
    check(not wrong_hash, f"all recorded sha256 match ({len(wrong_hash)} mismatched)")
    for lst, label in ((missing, "missing"), (wrong_size, "size"), (wrong_hash, "hash")):
        for x in lst[:5]:
            print(f"       {label}: {x}")

    on_disk = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    listed = {r["path"] for r in files} | {"MANIFEST.json"}
    untracked = sorted(on_disk - listed)
    check(not untracked, f"no untracked files ({len(untracked)} found)")
    for x in untracked[:5]:
        print(f"       untracked: {x}")
    stale = sorted(listed - on_disk)
    check(not stale, f"no stale manifest entries ({len(stale)} found)")


def check_no_figures() -> None:
    print("\n[2] bundle is data-only (no figures)")
    figs = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.suffix.lower() in FIGURE_SUFFIXES]
    check(not figs, f"no image files anywhere ({len(figs)} found)")
    for x in figs[:5]:
        print(f"       figure: {x}")
    refs = 0
    for name in ("README.md", "KNOWN_ISSUES.md"):
        text = (ROOT / name).read_text().lower()
        refs += sum(token in text for token in (".png", ".pdf", "figure", "plot"))
    check(True, f"docs mention plots {refs} time(s) (informational)")


def check_coverage() -> None:
    print("\n[3] per-cell coverage")
    cells = {f"{m}_seq{L}" for m in MODEL_TYPES for L in SEQUENCE_LENGTHS}
    check(len(cells) == 48, "48 expected cells")
    gen = {p.stem for p in (ROOT / "02_generated_outputs").glob("*/*.pt")}
    check(gen == cells, f"generated outputs for all 48 cells ({len(gen)}/48)")
    stats = {p.stem for p in (ROOT / "02_generated_outputs" / "block_bootstrap").glob("*.pt")}
    check(stats, "statistical generated outputs present (spot check)")
    refs = {p.stem for p in (ROOT / "03_ground_truth" / "reference").glob("*.npz")}
    check(cells <= refs, f"verified references for all 48 cells ({len(cells & refs)}/48)")
    gt = {p.stem for p in (ROOT / "03_ground_truth").glob("*.pt")}
    check(gt == {f"ground_truth_seq{L}" for L in SEQUENCE_LENGTHS}, f"canonical ground truth for 4 lengths ({len(gt)})")
    evalref = {p.stem for p in (ROOT / "03_ground_truth" / "eval_reference").glob("*.pt")}
    check(evalref == {f"ground_truth_seq{L}" for L in SEQUENCE_LENGTHS}, f"eval-reference ground truth for 4 lengths ({len(evalref)})")
    all_ck = [p.relative_to(ROOT).as_posix() for p in (ROOT / "01_model_checkpoints").rglob("*.pt")]
    check(all("/latest/" in f"/{p}" for p in all_ck),
          f"only the `latest` checkpoint set is shipped ({len(all_ck)} files)")
    rows = list(csv.DictReader((ROOT / "01_model_checkpoints" / "checkpoint_inventory.csv").open()))
    check(len(rows) == len(all_ck), f"checkpoint inventory lists exactly the shipped files ({len(rows)} rows / {len(all_ck)} files)")
    named = {Path(p).name for c in H.results().values() for p in (c.get("model_checkpoint_manifest") or [])}
    shipped = {Path(r["file"]).name for r in rows}
    check(bool(named) and named <= shipped,
          f"every checkpoint the results name is in the bundle ({len(named)} named, {len(named - shipped)} missing)")
    ref = [r for r in rows if r["referenced_by_results"] == "True"]
    check(len(ref) == len(named), f"inventory flags the same {len(named)} files as referenced_by_results ({len(ref)})")


def check_results() -> None:
    print("\n[4] results JSON")
    bundle = H.results()
    cells = {f"{m}_seq{L}" for m in MODEL_TYPES for L in SEQUENCE_LENGTHS}
    check(set(bundle) == cells, f"48 cells keyed by model x length ({len(bundle)})")
    required = {"FidelityEvaluator", "DiversityEvaluator", "StylizedFactsEvaluator", "utility", "portfolio", "pnl"}
    missing = sorted(c for c in bundle if not required <= set(bundle[c]))
    check(not missing, f"every cell has all 6 blocks ({len(missing)} incomplete)")

    bad_strategies = [c for c in bundle if len(next(iter(bundle[c]["portfolio"].values()))) != 9]
    check(not bad_strategies, f"every cell has 9 portfolio strategies ({len(bad_strategies)} odd)")
    real_keys = {k for c in bundle for na in bundle[c]["portfolio"] for k in bundle[c]["portfolio"][na]}
    found = sorted(k for k in real_keys if k.startswith("Real_GT_"))
    check(len(found) == 4, f"4 distinct real baselines: {found}")

    eq_1n = [
        c for c in bundle
        for na in bundle[c]["portfolio"]
        if (lambda w: all(abs(x - 1 / len(w)) < 1e-9 for x in w))(
            bundle[c]["portfolio"][na]["GMV_LongOnly"]["weights"]
        )
    ]
    check(not eq_1n, f"long-only GMV never collapses to 1/N ({len(eq_1n)} cases; shipped: 144)")
    turnovers = {v["turnover"] for c in bundle for na in bundle[c]["portfolio"] for v in bundle[c]["portfolio"][na].values()}
    check(turnovers == {None}, f"turnover is null everywhere ({turnovers})")

    worst_sortino = max(
        (abs(bundle[c]["pnl"][arm]["sortino_ratio"]) for c in bundle for arm in ("real", "synthetic")
         if isinstance(bundle[c]["pnl"][arm]["sortino_ratio"], (int, float))
         and abs(bundle[c]["pnl"][arm]["sortino_ratio"]) != float("inf")),
        default=0.0,
    )
    check(worst_sortino < 1e3, f"no epsilon-inflated Sortino (max |sortino| = {worst_sortino:.2f}; shipped 6.65e12)")
    check(all("n_sortino_undefined" in bundle[c]["pnl"][arm] for c in bundle for arm in ("real", "synthetic")),
          "every P&L arm reports its path counts")

    corrections = [c for c in bundle if "downstream_recomputed_with_fixed_evaluators" in bundle[c]]
    check(len(corrections) == 48, f"every cell records what was recomputed ({len(corrections)}/48)")


def check_alignment() -> None:
    print("\n[5] reference reconstruction + sample sizing")
    rows = H.alignment()
    check(len(rows) == 48, f"alignment table has 48 rows ({len(rows)})")
    cols = ["autocorr_returns.real", "autocorr_returns.synth",
            "volatility_clustering.real", "volatility_clustering.synth"]
    worst = max(float(r[c]) for r in rows for c in cols if r[c] != "")
    check(worst < 1e-6, f"reconstruction reproduces the recorded metrics (worst |delta| = {worst:.2e})")
    n252 = sorted({int(r["n_used_by_evaluator"]) for r in rows if int(r["sequence_length"]) == 252})
    check(max(n252) <= 9, f"effective n at L=252 <= 9 (observed {n252})")
    mismatch = 0
    for row in rows:
        model, L = row["model"], int(row["sequence_length"])
        ref = H.reference(model, L)
        gen = H.generated_outputs(model, L)
        if ref.shape[0] != int(row["n_used_by_evaluator"]) or min(ref.shape[0], gen.shape[0]) != ref.shape[0]:
            mismatch += 1
    check(mismatch == 0, f"reference row count == recorded n for all cells ({mismatch} mismatches)")


def check_loader() -> None:
    print("\n[6] loader API")
    model, L = "quantgan", 126
    check(H.generated_outputs(model, L).shape[1:] == (L, 25), "generated_outputs loads with the right shape")
    check(H.reference(model, L).shape[0] == H.effective_n(model, L), "reference rows == effective n")
    check(H.ground_truth(L).shape[1:] == (L, 25), "canonical ground truth loads")
    check(isinstance(H.cell_metadata(model, L), dict), "cell_metadata loads")
    check(isinstance(H.results_for(model, L), dict), "results_for returns the cell dict")
    check(H.metric(f"{model}_seq{L}", "pnl.synthetic.sharpe_ratio") == H.results_for(model, L)["pnl"]["synthetic"]["sharpe_ratio"],
          "dotted metric lookup matches the JSON")
    ckpt = H.checkpoint("quantgan")
    n = sum(1 for v in _flatten(ckpt) if hasattr(v, "shape"))
    check(n > 10, f"checkpoint loads ({n} tensors)")


def _flatten(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _flatten(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _flatten(v)
    else:
        yield node


def main() -> None:
    print(f"verifying bundle at {ROOT}")
    check_manifest()
    check_no_figures()
    check_coverage()
    check_results()
    check_alignment()
    check_loader()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
