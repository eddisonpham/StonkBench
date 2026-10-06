"""Move existing per-(model, seq_length) outputs to a versioned legacy bucket.

Before every retrain the pipeline calls :func:`archive_existing`, which scans
the ACTIVE run's ``outputs/results/{run_id}/{model}/`` and
``outputs/sanity/{run_id}/{model}/`` (plus run-scoped log files) for the
(model_key, generation_length) we're about to overwrite and relocates them to
a fresh ``outputs/_legacy/{UTC-iso-ts}_{model}_seq{L}/`` directory.

.. note::
   Scoped to the active run since 2026-10-05 (see
   ``_collect_active_sources``): an isolated-run retrain must never sweep
   another run's outputs (production ``results/latest`` was hit by the
   fair_20261004 retrains before this fix).

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


def _seq_name_matches(name: str, model_key: str, seq_lengths: "set[int]") -> bool:
    """True when ``name`` carries a ``_seq{L}`` token for one of ``seq_lengths``.

    The trailing separator check (``.`` or ``_`` after the number) keeps
    ``_seq2`` from matching ``_seq21``/``_seq252`` and vice versa.
    Files without any seq token are treated as model-level and are NOT
    matched (a seq-scoped retrain must not sweep them).
    """
    for length in seq_lengths:
        tok = f"_seq{length}"
        if name.startswith(f"{model_key}{tok}."):
            return True
        if f"{tok}." in name or f"{tok}_" in name:
            return True
    return False


def _collect_active_sources(
    output_root: Path,
    model_key: str,
    run_id: str | None = None,
    seq_lengths: "set[int] | None" = None,
) -> List[Path]:
    """Walk ``output_root`` for files owned by ``model_key`` in the ACTIVE
    run tree only (``run_id`` defaults to ``get_run_id()`` =
    ``$STONKBENCH_RUN_ID`` or ``"latest"``).

    Scoped 2026-10-05: the previous global scan across EVERY run tree moved
    other runs' outputs (e.g. production ``results/latest/<model>/``) into
    ``_legacy/`` whenever an isolated run retrained the same model — the
    fair_20261004 retrains swept all 7 production DL artifact sets this way.
    Now only ``results/{run_id}/{model}/``, ``sanity/{run_id}/{model}/`` and
    the run-scoped ``logs/{run_id}/**`` files of the model are archived. The
    append-only model-global ``logs/{model}/`` history is never touched.

    Seq scoping (2026-10-05, simulation_v2): when ``seq_lengths`` is given,
    results/sanity files are further filtered to exactly those sequence
    lengths — this is what the module docstring's "per-(model, seq_length)"
    always promised, and it makes PARALLEL per-seq retrains of the same
    model safe (a seq-21 task must not sweep the seq-126 artifacts another
    task wrote seconds earlier). ``None`` keeps the model-wide behaviour.
    Run-scoped logs stay model-level (append-only telemetry).
    """
    if not output_root.exists():
        return []
    if run_id is None:
        try:
            from src.utils.env import get_run_id

            run_id = get_run_id()
        except Exception:  # noqa: BLE001 — never fail archival on import issues
            run_id = "latest"

    sources: List[Path] = []
    # results/ and sanity/ have per-{run_id}/{model}/ shape — active run only.
    for top in ("results", "sanity"):
        mdir = output_root / top / run_id / model_key
        if not mdir.is_dir():
            continue
        for p in mdir.rglob("*"):
            if not p.is_file():
                continue
            if seq_lengths is not None:
                rel = p.relative_to(mdir)
                if top == "sanity":
                    if not any(part.startswith("seq") and part[3:].isdigit()
                               and int(part[3:]) in seq_lengths
                               for part in rel.parts):
                        continue
                elif not _seq_name_matches(p.name, model_key, seq_lengths):
                    continue
            sources.append(p)
    # Run-scoped logs only: files whose PATH COMPONENT exactly equals the
    # model key (2026-07-27 sweep: no substring over-match). The global
    # ``logs/{model_key}/`` tree is shared append-only history across runs
    # and must never be swept by an isolated run's retrain.
    logs_run = output_root / "logs" / run_id
    if logs_run.is_dir():
        for child in logs_run.rglob("*"):
            if not child.is_file():
                continue
            try:
                rel = child.relative_to(logs_run)
            except ValueError:
                continue
            if any(part == model_key for part in rel.parts):
                sources.append(child)
    return sources


def archive_existing(
    model_key: str,
    generation_length: int,
    output_root: Path,
    seq_lengths: "List[int] | None" = None,
) -> int:
    """Move active output for ``(model_key, seq_lengths or [generation_length])``
    to a fresh versioned subdir of ``output_root/_legacy/``.

    Parameters
    ----------
    model_key:
        directory immediately under ``results/{run_id}/``).
    generation_length:
        Sequence length being retrained. Names the legacy subdir
        (``{model_key}_seq{L}``) so a later grep over the legacy tree
        groups by (model, seq).
    output_root:
        StonkBench outputs root, normally ``outputs/`` (the value passed via
        ``--output_root`` to ``run_final_training``).
    seq_lengths:
        The full set of sequence lengths THIS run will regenerate (pipeline
        passes its ``seq_lengths``). Defaults to ``[generation_length]``.
        Only files belonging to those seqs are moved, so parallel per-seq
        tasks of the same model never sweep each other's outputs, while a
        multi-seq single-process run still evacuates everything it is about
        to rewrite.

    Returns
    -------
    int
        Number of files actually moved.
    """
    output_root = Path(output_root)
    seq_l = int(generation_length)
    seq_set = {int(x) for x in seq_lengths} if seq_lengths else {seq_l}
    sources = _collect_active_sources(
        output_root, model_key, seq_lengths=seq_set
    )
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
        "seq_lengths": sorted(seq_set),
        "files_moved": move_count,
        "buckets": bucket_counts,
        "legacy_path": str(legacy_root),
    }
    (legacy_root / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))
    return move_count
