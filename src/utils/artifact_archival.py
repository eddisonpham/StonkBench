"""Move existing per-(model, seq_length) outputs to a versioned legacy bucket.

Before every retrain the pipeline calls :func:`archive_existing`, which scans
``outputs/results/*/{model}/`` and ``outputs/sanity/*/{model}/`` (and the
matching log files) for the (model_key, generation_length) we're about to
overwrite and relocates them to a fresh
``outputs/_legacy/{UTC-iso-ts}_{model}_seq{L}/`` directory.

The active ``outputs/results/{run_id}/`` and ``outputs/sanity/{run_id}/`` are
then empty for that (model, seq), so the next retrain writes fresh items
without leaving a stale trail of "dated reruns" mixed into the live tree.

The move is idempotent:

* If there is nothing matching (model, seq) anywhere in the active tree, the
  helper returns ``0`` and writes nothing.
* If the legacy target folder already exists (two retrain requests landed
  within the same UTC second), the helper appends ``_attempt{N}`` to the
  candidate folder until a free slot is found.
* Individual file-name collisions inside the legacy bucket get a
  ``_attempt{N}`` suffix on the destination path before the move.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


def _ts_now() -> str:
    """UTC filename-safe timestamp ``YYYY-MM-DDTHH-MM-SSZ`` (no colons)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _next_free_legacy_root(legacy_root: Path) -> Path:
    """Append ``_attempt{N}`` until ``legacy_root`` does not yet exist."""
    if not legacy_root.exists():
        return legacy_root
    attempt = 2
    while True:
        candidate = legacy_root.parent / f"{legacy_root.name}_attempt{attempt}"
        if not candidate.exists():
            return candidate
        attempt += 1


def _destination_bucket(output_root: Path, src: Path) -> Tuple[Path, str]:
    """Classify ``src`` into (dest_subdir_under_legacy, hint_tag)."""
    try:
        rel = src.relative_to(output_root)
    except ValueError:
        # Files outside ``output_root`` are unlikely but defensively drop them
        # under the legacy root verbatim.
        return Path("."), "misc"
    parts = rel.parts
    if "artifacts" in parts:
        return Path("artifacts"), "artifacts"
    if "checkpoints" in parts:
        return Path("checkpoints"), "checkpoints"
    if "sanity" in parts:
        return Path("sanity"), "sanity"
    if "logs" in parts and "results" not in parts and "sanity" not in parts:
        return Path("logs"), "logs"
    return Path("other"), "other"


def _unique_destination(dest_dir: Path, name: str) -> Path:
    """Return ``dest_dir / name`` (or with ``_attempt{N}``) on collision."""
    candidate = dest_dir / name
    if not candidate.exists():
        return candidate
    stem, suf = candidate.stem, candidate.suffix
    attempt = 2
    while True:
        cand = dest_dir / f"{stem}_attempt{attempt}{suf}"
        if not cand.exists():
            return cand
        attempt += 1


def _collect_active_sources(
    output_root: Path, model_key: str
) -> List[Path]:
    """Walk ``output_root`` for every file owned by ``model_key`` that is NOT
    under ``_legacy/`` and is therefore eligible for archival on rerun."""
    if not output_root.exists():
        return []

    direct_subtrees = ("results", "sanity", "logs")
    sources: List[Path] = []
    for top in direct_subtrees:
        top_dir = output_root / top
        if not top_dir.exists():
            continue
        if top == "logs":
            # Logs are sometimes flat per-run, sometimes per-model.
            # Tightened 2026-07-27 (reviewer #3 sweep): only the canonical
            # events log (`{model_key}.jsonl`) AND files whose PATH
            # COMPONENT exactly equals the model key. Drop the substring
            # `parts contains model_key` over-match (e.g. a hypothetical
            # `cond_sig_wgan_v2_metrics.jsonl`) and the generic
            # `{model_key}.<ext>.log` filename prefix.
            for child in top_dir.rglob("*"):
                if not child.is_file():
                    continue
                if child.name == f"{model_key}.jsonl":
                    sources.append(child)
                    continue
                try:
                    rel = child.relative_to(output_root)
                except ValueError:
                    continue
                if any(part == model_key for part in rel.parts):
                    sources.append(child)
            continue
        # results/ and sanity/ have per-{run_id}/{model}/ shape
        for run_dir in top_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            mdir = run_dir / model_key
            if not mdir.exists():
                continue
            for p in mdir.rglob("*"):
                if p.is_file():
                    sources.append(p)
    return sources


def archive_existing(model_key: str, generation_length: int, output_root: Path) -> int:
    """Move every active output for ``(model_key, generation_length)`` to a
    fresh versioned subdir of ``output_root/_legacy/``.

    Parameters
    ----------
    model_key:
        Registry key, e.g. ``"cond_sig_wgan"`` or ``"pcf_gan"`` (matches the
        directory immediately under ``results/{run_id}/``).
    generation_length:
        Sequence length being retrained. Used to compose the legacy subdir
        name (``{model_key}_seq{L}``) so a later grep over the legacy tree
        groups by (model, seq).
    output_root:
        StonkBench outputs root, normally ``outputs/`` (the value passed via
        ``--output_root`` to ``run_final_training``).

    Returns
    -------
    int
        Number of files actually moved.
    """
    output_root = Path(output_root)
    seq_l = int(generation_length)
    sources = _collect_active_sources(output_root, model_key)
    if not sources:
        return 0

    ts = _ts_now()
    base_target = output_root / "_legacy" / f"{ts}_{model_key}_seq{seq_l}"
    legacy_root = _next_free_legacy_root(base_target)
    for sub in ("artifacts", "checkpoints", "sanity", "logs", "other"):
        (legacy_root / sub).mkdir(parents=True, exist_ok=True)

    move_count = 0
    bucket_counts: dict = {}
    for src in sources:
        bucket_dir, tag = _destination_bucket(output_root, src)
        dest_dir = legacy_root / bucket_dir
        dest = _unique_destination(dest_dir, src.name)
        try:
            shutil.move(str(src), str(dest))
        except OSError as exc:
            # Don't let one failed move derail the rest; the rest of the
            # active tree for this (model, seq) still ends up cleaned.
            print(f"[archive] failed to move {src} -> {dest}: {exc}")
            continue
        move_count += 1
        bucket_counts[tag] = bucket_counts.get(tag, 0) + 1

    manifest_payload = {
        "archived_at_utc": ts,
        "model_key": model_key,
        "generation_length": seq_l,
        "files_moved": move_count,
        "buckets": bucket_counts,
        "legacy_path": str(legacy_root),
    }
    (legacy_root / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))
    return move_count
