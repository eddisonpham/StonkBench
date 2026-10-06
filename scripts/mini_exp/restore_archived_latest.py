#!/usr/bin/env python
"""Restore production outputs that archive_existing swept across run trees.

Why this exists: ``archive_existing()`` scanned ``results/*`` and ``sanity/*``
for ALL run ids, so the fair_20261004 retrains moved the production
``results/latest/<model>/artifacts`` (n=9 / n=1000) and the model-scoped
event logs into ``_legacy/<ts>_<model>_seq<L>/`` buckets. This script puts
the production artifacts and event logs back — exactly, with verification:

  artifacts/<model>_seq{L}.pt  -> results/latest/<model>/artifacts/
      guarded by metadata.num_samples in {9, 1000} (production sizes) so it
      can never place a fair (n=500) or mini (n=200) file in results/latest.
  logs/events.jsonl            -> merged (bucket order by archive ts, then
      the current live file) into logs/<model>/events.jsonl, dedup by line.

Sanity PNGs in the buckets are deliberately NOT moved back: the flattening
(``hist_attemptN.png``) lost the seq{L}/{ticker}/ path, and readdir-order
reconstruction risks silently mislabelling plots. They stay preserved in
``_legacy/`` for manual retrieval; production sanity can be regenerated from
the restored artifacts if needed.

Only buckets dated this round (2026-10-04 onward) are touched — the August
bootstrap buckets are pre-existing legitimate archives. Dry-run by default;
pass --apply to move. Writes ``_legacy/restore_report.json``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

OUTPUT_ROOT = Path("/scratch/epham/stonkbench/output")
ROUND_CUTOFF = "2026-10-04"  # only buckets created by the fair-retrain round
PRODUCTION_NUM_SAMPLES = {9, 1000}
DL_MODELS = {
    "timegrad", "unconditional_tsdiffusion", "conditional_tsdiffusion",
    "quantgan", "cond_sig_wgan", "vrnn_klsched", "kalman_vae_safe",
}


def bucket_info(bucket: Path) -> tuple[str, str, str] | None:
    """Parse ``<ts>_<model>_seq<L>`` -> (ts, model, seq) or None."""
    name = bucket.name
    try:
        ts, rest = name.split("_", 1)
        # rest = <model>_seq<L>; model itself may contain underscores
        seq_part = rest.rsplit("_seq", 1)
        if len(seq_part) != 2:
            return None
        model, seq = seq_part
        datetime.strptime(ts[:19].replace("-", "").replace(":", ""), "%Y%m%dT%H%M%S")
        return ts, model, seq
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move files")
    args = ap.parse_args()

    legacy_root = OUTPUT_ROOT / "_legacy"
    buckets = []
    for b in sorted(legacy_root.iterdir()):
        if not b.is_dir():
            continue
        info = bucket_info(b)
        if info is None:
            continue
        ts, model, _seq = info
        if ts < ROUND_CUTOFF or model not in DL_MODELS:
            continue
        buckets.append((ts, model, b))
    buckets.sort(key=lambda x: (x[1], x[0]))  # by model, then ts

    report: dict = {"round": ROUND_CUTOFF, "artifacts_restored": [],
                    "artifacts_skipped": [], "logs_merged": {},
                    "sanity_left_in_legacy": {}, "errors": []}

    # ---- 1. artifacts ---------------------------------------------------
    for ts, model, bucket in buckets:
        art_dir = bucket / "artifacts"
        if not art_dir.is_dir():
            continue
        for pt in sorted(art_dir.glob("*.pt")):
            try:
                raw = torch.load(pt, map_location="cpu", weights_only=False)
                md = raw.get("metadata", {}) if isinstance(raw, dict) else {}
                n = md.get("num_samples")
            except Exception as exc:  # noqa: BLE001
                report["errors"].append(f"{pt}: load failed: {exc}")
                continue
            if n not in PRODUCTION_NUM_SAMPLES:
                report["artifacts_skipped"].append(
                    {"file": str(pt), "num_samples": n,
                     "reason": "not a production-sized artifact"})
                continue
            dest = OUTPUT_ROOT / "results" / "latest" / model / "artifacts" / pt.name
            if dest.exists():
                report["artifacts_skipped"].append(
                    {"file": str(pt), "num_samples": n,
                     "reason": f"dest already exists: {dest}"})
                continue
            if args.apply:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(pt), str(dest))
            report["artifacts_restored"].append(
                {"src": str(pt), "dest": str(dest), "num_samples": n,
                 "shape": list(raw["data"].shape) if isinstance(raw, dict)
                 and "data" in raw else None, "applied": args.apply})

    # ---- 2. event logs (merge in bucket-ts order, then live file) -------
    for model in sorted({m for _ts, m, _b in buckets}):
        parts: list[Path] = []
        for ts, m, bucket in buckets:
            if m == model:
                log = bucket / "logs" / "events.jsonl"
                if log.is_file():
                    parts.append(log)
        live = OUTPUT_ROOT / "logs" / model / "events.jsonl"
        if live.is_file():
            parts.append(live)
        if not parts:
            continue
        seen: set[str] = set()
        merged: list[str] = []
        for p in parts:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line and line not in seen:
                    seen.add(line)
                    merged.append(line)
        report["logs_merged"][model] = {
            "sources": [str(p) for p in parts], "lines": len(merged),
            "dest": str(live), "applied": args.apply}
        if args.apply:
            live.parent.mkdir(parents=True, exist_ok=True)
            tmp = live.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(merged) + "\n", encoding="utf-8")
            tmp.replace(live)

    # ---- 3. sanity: inventory only -------------------------------------
    for ts, model, bucket in buckets:
        san = bucket / "sanity"
        if san.is_dir():
            n = sum(1 for _ in san.rglob("*") if _.is_file())
            if n:
                report["sanity_left_in_legacy"][str(bucket)] = n

    out = legacy_root / "restore_report.json"
    if args.apply:
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"\n{'APPLIED' if args.apply else 'DRY-RUN'}: "
          f"{len(report['artifacts_restored'])} artifacts, "
          f"{len(report['logs_merged'])} logs, "
          f"{len(report['artifacts_skipped'])} skipped, "
          f"{len(report['errors'])} errors")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
