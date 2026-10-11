"""Unit tests for ``src.taxonomies.pnl``.

These ground the conventions that the downstream audit (``handover/KNOWN_ISSUES.md``)
found to be wrong in the shipped run:

* log returns compound as ``exp(cumsum(r))`` / ``exp(sum(r))`` — never ``cumprod(1+r)``;
* a path with no downside deviation has an *undefined* Sortino, which must be
  counted and excluded from the mean instead of being epsilon-patched to ~1e15.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from src.taxonomies.pnl import (  # noqa: E402  (needs the bootstrap first)
    PnLEvaluator,
    _accumulate_returns,
    _pnl_metrics,
)


class TestCompounding(unittest.TestCase):
    """Terminal wealth and cumulative paths must use the log-return convention."""

    def test_terminal_wealth_is_exp_of_log_return_sum(self):
        r = np.array([[0.01, -0.02, 0.03], [0.00, 0.00, 0.00]])
        m = _pnl_metrics(r)
        expected = np.exp(r.sum(axis=1))
        self.assertAlmostEqual(m["terminal_wealth_mean"], float(expected.mean()), places=12)
        self.assertAlmostEqual(m["terminal_wealth_min"], float(expected.min()), places=12)
        self.assertAlmostEqual(m["terminal_wealth_max"], float(expected.max()), places=12)

    def test_terminal_wealth_rejects_simple_return_compounding(self):
        # prod(1 + r) is the *simple*-return convention; on log returns it is wrong.
        r = np.array([[0.10, 0.10, 0.10]])
        log_convention = float(np.exp(r.sum(axis=1))[0])
        simple_convention = float(np.prod(1.0 + r[0]))
        got = _pnl_metrics(r)["terminal_wealth_mean"]
        self.assertAlmostEqual(got, log_convention, places=12)
        self.assertNotAlmostEqual(got, simple_convention, places=6)

    def test_accumulate_returns_matches_terminal_wealth(self):
        r = np.array([[0.02, -0.01, 0.03], [0.05, 0.01, -0.02]])
        path = _accumulate_returns(r, initial_value=1.0)
        self.assertTrue(np.allclose(path[:, -1], np.exp(r.sum(axis=1))))
        self.assertTrue(np.allclose(path, np.exp(np.cumsum(r, axis=1))))


class TestRatioMetrics(unittest.TestCase):
    """Ratios whose denominator vanishes are undefined, not huge."""

    def test_zero_downside_path_is_undefined_not_inflated(self):
        # One path with no loss at all: its Sortino is undefined.
        r = np.array([
            [0.01, 0.02, 0.03],   # no negative step -> undefined downside deviation
            [-0.01, 0.02, -0.02],
        ])
        m = _pnl_metrics(r)
        self.assertEqual(m["n_paths"], 2)
        self.assertEqual(m["n_sortino_undefined"], 1)
        self.assertEqual(m["n_sortino_defined"], 1)
        self.assertTrue(math.isfinite(m["sortino_ratio"]))
        self.assertLess(abs(m["sortino_ratio"]), 1e3)  # was 6.65e12 pre-fix

    def test_sortino_mean_excludes_undefined_paths(self):
        defined = np.array([[-0.01, 0.02, -0.02]])
        only_defined = _pnl_metrics(defined)["sortino_ratio"]
        with_undefined = _pnl_metrics(np.vstack([defined, [[0.01, 0.02, 0.03]]]))["sortino_ratio"]
        self.assertAlmostEqual(only_defined, with_undefined, places=12)

    def test_sortino_is_nan_when_every_path_is_undefined(self):
        m = _pnl_metrics(np.array([[0.01, 0.02, 0.03]]))
        self.assertEqual(m["n_sortino_defined"], 0)
        self.assertEqual(m["n_sortino_undefined"], 1)
        self.assertTrue(math.isnan(m["sortino_ratio"]))

    def test_omega_infinite_paths_are_counted(self):
        m = _pnl_metrics(np.array([[0.01, 0.02, 0.03]]))  # no losses -> omega undefined
        self.assertEqual(m["n_omega_infinite"], 1)
        self.assertTrue(math.isinf(m["omega_ratio"]))

    def test_sharpe_on_known_returns(self):
        r = np.array([[0.01, -0.01, 0.01, -0.01]])
        m = _pnl_metrics(r, periods_per_year=252)
        std = np.std(r[0], ddof=1)
        self.assertAlmostEqual(m["sharpe_ratio"], (r[0].mean() / std) * math.sqrt(252), places=12)

    def test_max_drawdown_on_known_path(self):
        # wealth: 1.00 -> 1.10 -> 0.99 -> 1.20  => worst peak-to-trough is -10%
        r = np.array([[math.log(1.10), math.log(0.9), math.log(1.2 / 0.9)]])
        m = _pnl_metrics(r)
        self.assertAlmostEqual(m["max_drawdown"], -0.10, places=12)

    def test_prob_positive_return_counts_paths(self):
        r = np.array([[0.01, 0.01], [-0.01, -0.01], [0.01, -0.005]])
        m = _pnl_metrics(r)
        self.assertAlmostEqual(m["prob_positive_return"], 2.0 / 3.0, places=12)


class TestPnLEvaluator(unittest.TestCase):
    """The evaluator must compare the two arms on the same conventions."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.real = rng.normal(0.001, 0.01, size=(6, 21, 5)).astype(np.float64)
        self.synth = rng.normal(0.001, 0.01, size=(6, 21, 5)).astype(np.float64)

    def test_output_shape_and_keys(self):
        out = PnLEvaluator(self.real, self.synth).evaluate()
        for key in ("real", "synthetic", "terminal_wealth_wasserstein", "strategy", "strategy_weights"):
            self.assertIn(key, out)
        self.assertEqual(out["strategy"], "equal_weight_bh")
        self.assertEqual(len(out["strategy_weights"]), 5)
        self.assertAlmostEqual(sum(out["strategy_weights"]), 1.0, places=12)

    def test_wasserstein_is_zero_for_identical_arms(self):
        out = PnLEvaluator(self.real, self.real).evaluate()
        self.assertAlmostEqual(out["terminal_wealth_wasserstein"], 0.0, places=12)

    def test_arms_keep_their_own_path_counts(self):
        out = PnLEvaluator(self.real, self.synth).evaluate()
        self.assertEqual(out["real"]["n_paths"], 6)
        self.assertEqual(out["synthetic"]["n_paths"], 6)

    def test_accepts_2d_input(self):
        out = PnLEvaluator(self.real[0], self.synth[0]).evaluate()
        self.assertEqual(out["real"]["n_paths"], 1)
        self.assertEqual(out["synthetic"]["n_paths"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
