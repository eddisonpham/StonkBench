"""Build ``EVAL_INPUTS.json`` — the index of everything the evaluation stage consumes.

Scans the multi-batch generation tree and records one entry per
(ablation, batch, model, seq) cell — reading the per-cell ``gen_seq{L}.json``
sidecars (falling back to artifact metadata) — plus the ground-truth
references, so an eval job (or an auditor) can enumerate inputs without
walking the tree by hand.

Outputs (next to the tree root)::

    <root>/EVAL_INPUTS.json     full index + counts + ready-to-run eval commands
    <root>/EVAL_INPUTS.csv      flat table for spreadsheets

Usage:
    python scripts/mini_exp/build_eval_manifest.py \
        --root /home/epham/StonkBench/outputs/generation
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS = [
    "unconditional_tsdiffusion",
    "conditional_tsdiffusion",
    "quantgan",
    "timegrad",
    "vrnn_klsched",
    "kalman_vae_safe",
    "cond_sig_wgan",
]
STAT_MODELS = [
    "block_bootstrap",
    "stationary_block_bootstrap",
    "merton_jump_diffusion",
    "de_jump_diffusion",
    "dcc_garch",
]
SEQS = [21, 42, 126, 252]
BATCHES = [1, 2, 3]
ABLATIONS = ["simulation_v2", "fair_20261004", "latest"]


def _read_sidecar(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _read_artifact(path: Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["metadata"]
    data = payload["data"]
    return {
        "shape": list(data.shape),
        "std": float(data.std()),
        "max_abs": float(data.abs().max()),
        "seed": int(meta.get("seed", -1)),
        "batch": meta.get("generation_batch"),
        "native_length": meta.get("native_length"),
        "model_family": meta.get("model_family"),
        "model_checkpoint_manifest": meta.get("model_checkpoint_manifest"),
        "restored_from_checkpoint": meta.get("restored_from_checkpoint"),
        "fit_from_stats_set": meta.get("fit_from_stats_set"),
        "stitched_from_native": meta.get("stitched_from_native"),
        "path": str(path),
    }


def build(root: Path) -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    missing: List[str] = []

    for ablation in ABLATIONS:
        for batch in BATCHES:
            for model in MODELS + STAT_MODELS:
                family = "statistical" if model in STAT_MODELS else "deep_learning"
                for seq in SEQS:
                    cell_dir = root / ablation / f"batch{batch}" / model
                    art = cell_dir / "artifacts" / f"{model}_seq{seq}.pt"
                    side = cell_dir / f"gen_seq{seq}.json"
                    if not art.exists():
                        missing.append(str(art.relative_to(root)))
                        continue
                    record = _read_sidecar(side) if side.exists() else None
                    info = record if record else _read_artifact(art)
                    cells.append(
                        {
                            "ablation": ablation,
                            "batch": batch,
                            "model": model,
                            "model_family": family,
                            "sequence_length": seq,
                            "path": str(art),
                            "relative_path": str(art.relative_to(root)),
                            "sidecar": str(side) if side.exists() else None,
                            "seed": info.get("seed"),
                            "num_samples": info.get("num_samples")
                            or (info.get("shape") or [None])[0],
                            "shape": info.get("shape"),
                            "std": info.get("std"),
                            "native_length": info.get("native_length"),
                            "checkpoint": info.get("checkpoint")
                            or (info.get("model_checkpoint_manifest") or [None] or [None])[0],
                            "ground_truth_dl": str(
                                root / ablation / "ground_truth" / f"ground_truth_seq{seq}.pt"
                            ),
                            "ground_truth_statistical": str(
                                root / ablation / "ground_truth" / "statistical"
                                / f"ground_truth_seq{seq}.pt"
                            ),
                        }
                    )

    counts: Dict[str, Any] = {}
    for ablation in ABLATIONS:
        sub = [c for c in cells if c["ablation"] == ablation]
        counts[ablation] = {
            "cells": len(sub),
            "deep_learning": len([c for c in sub if c["model_family"] == "deep_learning"]),
            "statistical": len([c for c in sub if c["model_family"] == "statistical"]),
            "batches": sorted({c["batch"] for c in sub}),
            "models": sorted({c["model"] for c in sub}),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "ablations": ABLATIONS,
        "batches": BATCHES,
        "seq_lengths": SEQS,
        "samples_per_batch": 500,
        "batch_seeds": {str(b): 42 + 1000 * b for b in BATCHES},
        "totals": {
            "expected_cells": len(ABLATIONS) * len(BATCHES) * len(MODELS + STAT_MODELS) * len(SEQS),
            "indexed_cells": len(cells),
            "missing": missing,
        },
        "per_ablation": counts,
        "cells": cells,
        "eval": {
            "staged_script": "scripts/mini_exp/eval_gen_batches.sh",
            "status": (
                "RUNNING — 1040043 (array 0-431) + 1040158 (array 24-431, "
                "2h backfill re-submission) + 1040509 (timeout requeue, cell 26) "
                "+ 1044081 (top-up cells 153,154,157,159); finisher 1044082 "
                "chained afterany on 1040158/1040509/1044081"
            ),
            "cell_map": (
                "tid%4 -> seq index into [21,42,126,252]; (tid/4)%12 -> model; "
                "(tid/48)%3+1 -> batch; tid/144 -> ablation"
            ),
            "aggregator": "scripts/mini_exp/aggregate_batch_metrics.py",
            "commands": {
                ablation: (
                    f"python -m src.unified_evaluator "
                    f"--generated_dir <root>/{ablation}/batch1 "
                    f"--results_dir <root>/{ablation}/evaluation_batch1/block_bootstrap_seq21 "
                    f"--model block_bootstrap --seq_length 21 --skip_regenerate"
                )
                for ablation in ABLATIONS
            },
            "note": (
                "One artifact per array task, each with its OWN results_dir "
                "(<root>/<ablation>/evaluation_batch<k>/<model>_seq<L>/) because the "
                "evaluator OVERWRITES complete_evaluation.json per run; the "
                "aggregator reads those per-cell summaries directly, so there is no "
                "shard-merge step. Every task also writes its cell's plots. No "
                "--seq_lengths (plural) filter: that would trim the last L steps. "
                "--model is matched EXACTLY (a substring test would let "
                "block_bootstrap pick up stationary_block_bootstrap). Real-data side "
                "resolves from $STONKBENCH_DL_SET_PATH (dl_set.pt = W252), matching "
                "the accepted simulation_v2 protocol."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=str(REPO_ROOT / "outputs" / "generation"),
        help="generation root (default <repo>/outputs/generation)",
    )
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[manifest] root missing: {root}")
        return 2

    manifest = build(root)
    json_path = root / "EVAL_INPUTS.json"
    csv_path = root / "EVAL_INPUTS.csv"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=False, default=str))
    with csv_path.open("w", newline="") as fh:
        fields = [
            "ablation", "batch", "model", "model_family", "sequence_length",
            "seed", "num_samples", "native_length", "std", "relative_path",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest["cells"])

    t = manifest["totals"]
    print(f"[manifest] wrote {json_path}\n           {csv_path}")
    print(
        f"[manifest] indexed {t['indexed_cells']}/{t['expected_cells']} cells "
        f"({len(t['missing'])} missing)"
    )
    for ablation, info in manifest["per_ablation"].items():
        print(
            f"  {ablation}: {info['cells']} cells "
            f"({info['deep_learning']} DL + {info['statistical']} stat) "
            f"batches={info['batches']} models={len(info['models'])}"
        )
    if t["missing"]:
        for m in t["missing"][:20]:
            print(f"  MISSING {m}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
