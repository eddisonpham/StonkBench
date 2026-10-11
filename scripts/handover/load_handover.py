#!/usr/bin/env python3
"""Read-side API for the handover bundle.

Only ``numpy`` is required (``torch_pt_reader`` reads the ``.pt`` artifacts
without torch)::

    import sys; sys.path.insert(0, "scripts/handover")
    from load_handover import Handover

    h = Handover()                             # auto-locates the bundle
    h.cells()                                  # 48 cell ids
    X = h.generated_outputs("quantgan", 126)   # (9, 126, 25) float32
    R = h.reference("quantgan", 126)           # the real data that cell was scored on
    h.effective_n("quantgan", 126)             # 9
    h.results_for("quantgan", 126)             # the full per-cell evaluation dict
    h.metadata()                               # alignment / directions / fix list

Every accessor raises ``KeyError`` with the available keys on a bad lookup.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from torch_pt_reader import load_pt

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parents[1]
DEFAULT_ROOT = REPO / "handover"

MODEL_TYPES = {
    "block_bootstrap": "statistical",
    "dcc_garch": "statistical",
    "de_jump_diffusion": "statistical",
    "merton_jump_diffusion": "statistical",
    "stationary_block_bootstrap": "statistical",
    "cond_sig_wgan": "deep_learning",
    "conditional_tsdiffusion": "deep_learning",
    "kalman_vae_safe": "deep_learning",
    "quantgan": "deep_learning",
    "timegrad": "deep_learning",
    "unconditional_tsdiffusion": "deep_learning",
    "vrnn_klsched": "deep_learning",
}
SEQUENCE_LENGTHS = (21, 42, 126, 252)
RESULTS_DIR = "04_evaluation_results"


@dataclass(frozen=True)
class Cell:
    model: str
    sequence_length: int

    @property
    def model_type(self) -> str:
        return MODEL_TYPES[self.model]

    @property
    def key(self) -> str:
        return f"{self.model}_seq{self.sequence_length}"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.key


class Handover:
    """Lazy view over ``handover/``."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else DEFAULT_ROOT
        if not (self.root / "MANIFEST.json").exists():
            raise FileNotFoundError(f"{self.root} does not look like the handover bundle (no MANIFEST.json)")
        self._manifest = None
        self._results = None
        self._metadata = None

    # ------------------------------------------------------------------ #
    # inventory
    # ------------------------------------------------------------------ #
    def manifest(self) -> dict:
        """``MANIFEST.json`` — every file with path, bytes, sha256 and kind."""
        if self._manifest is None:
            self._manifest = json.loads((self.root / "MANIFEST.json").read_text())
        return self._manifest

    def results(self) -> dict:
        """``{cell_id: {...}}`` — the whole evaluation, all 48 cells, all taxonomies."""
        if self._results is None:
            self._results = json.loads((self.root / RESULTS_DIR / "complete_evaluation.json").read_text())
        return self._results

    def metadata(self) -> dict:
        """Alignment table, metric directions, and the list of applied fixes."""
        if self._metadata is None:
            self._metadata = json.loads((self.root / RESULTS_DIR / "evaluation_metadata.json").read_text())
        return self._metadata

    def files(self, kind: str | None = None) -> list[dict]:
        """Manifest rows, optionally filtered by kind."""
        recs = self.manifest()["files"]
        return [r for r in recs if kind is None or r.get("kind") == kind]

    def cells(self) -> list[Cell]:
        return [Cell(m, L) for m in MODEL_TYPES for L in SEQUENCE_LENGTHS]

    def cell(self, model: str, sequence_length: int) -> Cell:
        if model not in MODEL_TYPES:
            raise KeyError(f"unknown model {model!r}; known: {sorted(MODEL_TYPES)}")
        if sequence_length not in SEQUENCE_LENGTHS:
            raise KeyError(f"unknown sequence_length {sequence_length!r}; known: {SEQUENCE_LENGTHS}")
        return Cell(model, sequence_length)

    # ------------------------------------------------------------------ #
    # raw data
    # ------------------------------------------------------------------ #
    def generated_outputs(self, model: str, sequence_length: int):
        """Generated samples ``(n_samples, L, 25)`` for one cell."""
        c = self.cell(model, sequence_length)
        path = self.root / "02_generated_outputs" / model / f"{c.key}.pt"
        return load_pt(path)["data"]

    def ground_truth(self, sequence_length: int, variant: str = "canonical"):
        """Ground-truth windows ``(n_windows, L, 25)``.

        ``variant='canonical'`` is the full window set under ``outputs/ground_truth``;
        ``variant='eval_reference'`` is the smaller set the evaluation directory held.
        NB neither is what the evaluator scored against — use :meth:`reference`.
        """
        if sequence_length not in SEQUENCE_LENGTHS:
            raise KeyError(f"unknown sequence_length {sequence_length!r}; known: {SEQUENCE_LENGTHS}")
        if variant == "eval_reference":
            path = self.root / "03_ground_truth" / "eval_reference" / f"ground_truth_seq{sequence_length}.pt"
        elif variant == "canonical":
            path = self.root / "03_ground_truth" / f"ground_truth_seq{sequence_length}.pt"
        else:
            raise KeyError("variant must be 'canonical' or 'eval_reference'")
        return load_pt(path)["data"]

    def reference(self, model: str, sequence_length: int):
        """The verified real-data reference this cell was actually scored against.

        Reconstructed from ``dl_set_W252.test_series_raw`` by
        ``reconstruct_references.py`` using the evaluator's own rule: the whole
        stride-1 window pool for statistical models, its second half for deep
        learning models, truncated to the aligned sample count.
        """
        c = self.cell(model, sequence_length)
        path = self.root / "03_ground_truth" / "reference" / f"{c.key}.npz"
        if not path.exists():
            raise KeyError(f"no reconstructed reference for {c.key}")
        with np.load(path) as z:
            return z["data"]

    def effective_n(self, model: str, sequence_length: int) -> int:
        """How many real windows (and synthetic samples) the cell was scored on."""
        for row in self.metadata()["alignment"]:
            if row["model"] == model and int(row["sequence_length"]) == sequence_length:
                return int(row["n_used_by_evaluator"])
        raise KeyError(f"no alignment row for {model} seq{sequence_length}")

    def alignment(self) -> list[dict]:
        """Per-cell sample counts, reference pool, and the reconstruction proof deltas."""
        return self.metadata()["alignment"]

    def cell_metadata(self, model: str, sequence_length: int) -> dict:
        path = self.root / "02_generated_outputs" / model / f"{self.cell(model, sequence_length).key}.pt"
        return load_pt(path)["metadata"]

    def checkpoint_inventory(self) -> list[dict]:
        """One row per shipped checkpoint: model, file, bytes, sha256, referenced_by_results."""
        return self.read_csv(self.root / "01_model_checkpoints" / "checkpoint_inventory.csv")

    @staticmethod
    def read_csv(path: Path) -> list[dict]:
        with Path(path).open(newline="") as fh:
            return list(csv.DictReader(fh))

    def checkpoint_path(self, model: str, which: str = "referenced") -> Path:
        """Path to a trained checkpoint.

        Only the ``latest`` set is shipped — the set the results'
        ``model_checkpoint_manifest`` points into.  ``which='referenced'`` returns
        exactly that file; ``which='any'`` returns any checkpoint shipped for the
        model (some adapters keep a legacy-named copy alongside it).
        """
        rows = [r for r in self.checkpoint_inventory() if r["model"] == model]
        if not rows:
            kind = MODEL_TYPES.get(model)
            hint = "statistical models are fitted per generation and have no checkpoint" if kind == "statistical" else ""
            raise KeyError(f"no checkpoint shipped for {model!r}; {hint}".rstrip("; "))
        if which == "referenced":
            rows = [r for r in rows if r["referenced_by_results"] == "True"] or rows
        elif which != "any":
            raise KeyError("which must be 'referenced' or 'any'")
        return self.root / "01_model_checkpoints" / rows[0]["file"]

    def checkpoint(self, model: str, which: str = "referenced"):
        """Load a trained checkpoint (see :meth:`checkpoint_path`)."""
        return load_pt(self.checkpoint_path(model, which))

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    def results_for(self, model: str, sequence_length: int) -> dict:
        """The full per-cell evaluation dict (all taxonomies + downstream)."""
        key = self.cell(model, sequence_length).key
        try:
            return self.results()[key]
        except KeyError:
            raise KeyError(f"no results for {key}; known: {sorted(self.results())[:3]}...") from None

    def metric(self, cell_id: str, path: str):
        """Dotted lookup, e.g. ``metric("quantgan_seq126", "pnl.synthetic.sharpe_ratio")``."""
        node = self.results()[cell_id]
        for part in path.split("."):
            node = node[part]
        return node

    def alignment_csv(self) -> list[dict]:
        """The tool-side alignment table (same rows as ``metadata()['alignment']``)."""
        with (CODE_DIR / "data" / "evaluation_alignment.csv").open(newline="") as fh:
            return list(csv.DictReader(fh))


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    h = Handover()
    print(f"bundle root      : {h.root}")
    print(f"files in manifest: {len(h.manifest()['files'])}  ({h.manifest()['total_bytes'] / 1048576:.1f} MB)")
    print(f"cells            : {len(h.results())}")
    for c in h.cells()[:3]:
        print(f"  {c.key}: generated={h.generated_outputs(c.model, c.sequence_length).shape} "
              f"reference={h.reference(c.model, c.sequence_length).shape} n={h.effective_n(c.model, c.sequence_length)}")
    print("quantgan_seq126 pnl.synthetic.sharpe_ratio =",
          h.metric("quantgan_seq126", "pnl.synthetic.sharpe_ratio"))
