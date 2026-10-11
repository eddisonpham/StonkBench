"""Unit tests for the handover bundle reader (``scripts/handover/load_handover.py``).

The bundle is regenerable data, so these tests skip when ``handover/`` is absent
(a fresh clone will not have it).  When it is present they assert the invariants
that make the results trustworthy: 48 cells, per-cell sample alignment, the
corrected downstream blocks, and manifest integrity.
"""

from __future__ import annotations

import hashlib
import math
import unittest
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from load_handover import MODEL_TYPES, SEQUENCE_LENGTHS, Handover  # noqa: E402

BUNDLE = Path(__file__).resolve().parents[1] / "handover"
HAS_BUNDLE = (BUNDLE / "MANIFEST.json").exists()

CELLS = [f"{m}_seq{L}" for m in MODEL_TYPES for L in SEQUENCE_LENGTHS]


@unittest.skipUnless(HAS_BUNDLE, "handover/ bundle is not present (regenerate with scripts/handover/)")
class TestResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = Handover()

    def test_48_cells_with_every_block(self):
        results = self.h.results()
        self.assertEqual(sorted(results), sorted(CELLS))
        required = {"FidelityEvaluator", "DiversityEvaluator", "StylizedFactsEvaluator",
                    "utility", "portfolio", "pnl"}
        for cell_id, cell in results.items():
            self.assertTrue(required <= set(cell), msg=cell_id)

    def test_dotted_metric_lookup_matches_the_json(self):
        cell = self.h.results_for("quantgan", 126)
        self.assertEqual(self.h.metric("quantgan_seq126", "pnl.synthetic.sharpe_ratio"),
                         cell["pnl"]["synthetic"]["sharpe_ratio"])

    def test_unknown_keys_raise_key_error(self):
        with self.assertRaises(KeyError):
            self.h.results_for("no_such_model", 21)
        with self.assertRaises(KeyError):
            self.h.generated_outputs("quantgan", 999)

    def test_corrected_downstream_blocks(self):
        for cell_id, cell in self.h.results().items():
            portfolio = cell["portfolio"]
            keys = {k for strategies in portfolio.values() for k in strategies}
            self.assertEqual(len([k for k in keys if k.startswith("Real_GT_")]), 4, msg=cell_id)
            for strategies in portfolio.values():
                self.assertEqual(len(strategies), 9, msg=cell_id)
                for name, res in strategies.items():
                    self.assertIsNone(res["turnover"], msg=f"{cell_id}/{name}")
            for arm in ("real", "synthetic"):
                sortino = cell["pnl"][arm]["sortino_ratio"]
                self.assertTrue(math.isnan(sortino) or abs(sortino) < 1e3,
                                msg=f"{cell_id}/{arm} sortino={sortino}")

    def test_every_cell_records_what_was_recomputed(self):
        for cell_id, cell in self.h.results().items():
            self.assertIn("downstream_recomputed_with_fixed_evaluators", cell, msg=cell_id)


@unittest.skipUnless(HAS_BUNDLE, "handover/ bundle is not present (regenerate with scripts/handover/)")
class TestDataAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = Handover()

    def test_reference_rows_equal_effective_n(self):
        for cell in self.h.cells():
            ref = self.h.reference(cell.model, cell.sequence_length)
            self.assertEqual(ref.shape[0], self.h.effective_n(cell.model, cell.sequence_length),
                             msg=cell.key)

    def test_generated_output_shapes(self):
        for cell in self.h.cells():
            X = self.h.generated_outputs(cell.model, cell.sequence_length)
            self.assertEqual(X.shape[1:], (cell.sequence_length, 25), msg=cell.key)

    def test_alignment_table_has_proof_deltas(self):
        rows = self.h.alignment()
        self.assertEqual(len(rows), 48)
        worst = max(float(r[c]) for r in rows
                    for c in ("autocorr_returns.real", "volatility_clustering.real") if r[c])
        self.assertLess(worst, 1e-6, "reference reconstruction must reproduce the recorded metrics")

    def test_l252_is_starved_of_real_windows(self):
        n_at_252 = {self.h.effective_n(c.model, 252) for c in self.h.cells()}
        self.assertLessEqual(max(n_at_252), 9)

    def test_statistics_models_have_no_checkpoint(self):
        with self.assertRaises(KeyError):
            self.h.checkpoint("block_bootstrap")

    def test_checkpoint_inventory_matches_shipped_files(self):
        rows = self.h.checkpoint_inventory()
        self.assertEqual(len(rows), 10)
        for row in rows:
            path = BUNDLE / "01_model_checkpoints" / row["file"]
            self.assertTrue(path.exists(), msg=row["file"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, row["sha256"], msg=row["file"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
