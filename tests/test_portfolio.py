"""Unit tests for ``src.taxonomies.portfolio``.

These ground the defects documented in ``handover/KNOWN_ISSUES.md``:

* the long-only GMV optimiser must not stall at its 1/N starting point;
* the four real-data baselines must keep distinct keys (the shipped run collided
  them onto two, so the "oracle" was really the long-only run);
* ``turnover`` is undefined without a weight history — ``None``, not ``0.0``;
* log returns compound as ``exp(cumsum(r))``.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from src.taxonomies.portfolio import (  # noqa: E402  (needs the bootstrap first)
    PortfolioOptimizationEvaluator,
    _minimum_variance_weights,
    _portfolio_metrics,
    _sample_covariance,
    _tangency_weights,
)

try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except Exception:  # pragma: no cover - environment without scipy
    HAS_SCIPY = False


class TestWeights(unittest.TestCase):
    """Weight solvers must return feasible, sensible portfolios."""

    def test_unconstrained_gmv_matches_closed_form(self):
        cov = np.array([[0.04, 0.001], [0.001, 0.09]])
        got = _minimum_variance_weights(cov, allow_short=True)
        inv = np.linalg.pinv(cov)
        expected = (inv @ np.ones((2, 1))).flatten()
        expected = expected / expected.sum()
        self.assertTrue(np.allclose(got, expected, atol=1e-10))

    def test_unconstrained_gmv_can_short(self):
        # A negatively correlated pair makes shorting worthwhile.
        cov = np.array([[1.0, -0.9], [-0.9, 1.0]])
        w = _minimum_variance_weights(cov, allow_short=True)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=10)
        self.assertTrue((w[0] * w[1] < 0).any() or np.allclose(w, [0.5, 0.5]))

    def test_weights_are_feasible(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0, 0.01, size=(500, 6))
        cov = _sample_covariance(returns)
        mean = returns.mean(axis=0)
        for w in (
            _minimum_variance_weights(cov, allow_short=True),
            _minimum_variance_weights(cov, allow_short=False),
            _tangency_weights(cov, mean, allow_short=True),
            _tangency_weights(cov, mean, allow_short=False),
        ):
            self.assertAlmostEqual(float(np.sum(w)), 1.0, places=8)
        for w in (_minimum_variance_weights(cov, allow_short=False),
                  _tangency_weights(cov, mean, allow_short=False)):
            self.assertTrue((w >= -1e-12).all(), "long-only weights must be non-negative")

    @unittest.skipUnless(HAS_SCIPY, "scipy is required for the constrained optimisers")
    def test_long_only_gmv_beats_equal_weights(self):
        # Diagonal covariance: the optimum is w ∝ 1/σ² = (0.8, 0.2), strictly
        # better than 1/N.  The pre-fix solver returned exactly 1/N here.
        cov = np.diag([1.0, 4.0])
        w = _minimum_variance_weights(cov, allow_short=False)
        self.assertFalse(np.allclose(w, [0.5, 0.5]), "long-only GMV stalled at 1/N")
        self.assertTrue(float(w @ cov @ w) < float(np.array([0.5, 0.5]) @ cov @ np.array([0.5, 0.5])))

    @unittest.skipUnless(HAS_SCIPY, "scipy is required for the constrained optimisers")
    def test_long_only_gmv_does_not_stall_on_rescaled_objective(self):
        # Tiny covariance values used to make SLSQP's relative test pass at the
        # starting point; the objective is rescaled to O(1) to avoid that.
        cov = np.diag([1.26e-4, 4.0e-5, 2.0e-4, 8.0e-5, 1.0e-4])
        eq = np.ones(5) / 5
        w = _minimum_variance_weights(cov, allow_short=False)
        self.assertLess(float(w @ cov @ w), float(eq @ cov @ eq) - 1e-18)


class TestPortfolioMetrics(unittest.TestCase):
    """Metric conventions must match ``src.taxonomies.pnl``."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.r = rng.normal(0.0005, 0.01, size=300)

    def test_compounding_uses_log_returns(self):
        m = _portfolio_metrics(self.r)
        cumulative = np.exp(np.cumsum(self.r))
        hwm = np.maximum.accumulate(cumulative)
        expected = float(np.min((cumulative - hwm) / hwm))
        self.assertAlmostEqual(m["max_drawdown"], expected, places=12)

    def test_turnover_is_none_without_weight_history(self):
        w = np.ones(5) / 5
        m = _portfolio_metrics(self.r, weights_history=w[np.newaxis, :])
        self.assertIsNone(m["turnover"])

    def test_turnover_computed_from_weight_history(self):
        hist = np.array([[0.2, 0.2, 0.2, 0.2, 0.2], [0.4, 0.3, 0.1, 0.1, 0.1]])
        m = _portfolio_metrics(self.r, weights_history=hist)
        self.assertIsNotNone(m["turnover"])
        self.assertGreater(m["turnover"], 0.0)

    def test_herfindahl_from_last_weights(self):
        hist = np.array([[0.2, 0.2, 0.2, 0.2, 0.2], [0.4, 0.3, 0.1, 0.1, 0.1]])
        m = _portfolio_metrics(self.r, weights_history=hist)
        self.assertAlmostEqual(m["herfindahl"], float((hist[-1] ** 2).sum()), places=12)

    def test_zero_downside_returns_is_undefined_not_zero(self):
        r = np.full(50, 0.001)  # no losses at all
        m = _portfolio_metrics(r)
        self.assertTrue(math.isinf(m["sortino_ratio"]))

    def test_short_series_returns_empty(self):
        self.assertEqual(_portfolio_metrics(np.array([0.1])), {})


class TestEvaluator(unittest.TestCase):
    """End-to-end: strategies, baselines and the asset-count ablation."""

    def setUp(self):
        rng = np.random.default_rng(11)
        self.real = rng.normal(0.0004, 0.01, size=(252, 6))
        self.synth = rng.normal(0.0004, 0.01, size=(300, 6))

    def test_ablation_keys_and_strategy_keys(self):
        out = PortfolioOptimizationEvaluator(
            self.real, self.synth, num_assets_ablation=[3, 6]
        ).evaluate()
        self.assertEqual(sorted(out), ["n_assets_3", "n_assets_6"])
        expected = {
            "1N", "GMV", "GMV_LongOnly", "Tangency", "Tangency_LongOnly",
            "Real_GT_RealGMV", "Real_GT_RealGMV_LongOnly",
            "Real_GT_RealTangency", "Real_GT_RealTangency_LongOnly",
        }
        for strategies in out.values():
            self.assertEqual(set(strategies), expected)

    def test_real_baselines_are_four_distinct_keys(self):
        # The shipped run built the keys with str.split('_'), which mapped both
        # the oracle and the long-only variant onto the same name: only 2 of the
        # 4 baselines survived and the oracle silently held the long-only run.
        out = PortfolioOptimizationEvaluator(
            self.real, self.synth, num_assets_ablation=[6]
        ).evaluate()["n_assets_6"]
        keys = [k for k in out if k.startswith("Real_GT_")]
        self.assertEqual(
            sorted(keys),
            ["Real_GT_RealGMV", "Real_GT_RealGMV_LongOnly",
             "Real_GT_RealTangency", "Real_GT_RealTangency_LongOnly"],
        )
        for key in keys:
            self.assertEqual(len(out[key]["weights"]), 6, msg=key)

    def test_real_oracle_and_long_only_differ_when_shorting_helps(self):
        # A high-volatility asset that is highly correlated with a low-volatility
        # one makes the unconstrained GMV short the former (w1 < 0), which the
        # long-only variant cannot do -- so the two baselines must disagree.
        cov = np.array([[0.04, 0.018], [0.018, 0.010]])  # sigma = (0.2, 0.1), rho = 0.9
        rng = np.random.default_rng(3)
        real = rng.normal(size=(4000, 2)) @ np.linalg.cholesky(cov).T
        out = PortfolioOptimizationEvaluator(
            real, real, num_assets_ablation=[2]
        ).evaluate()["n_assets_2"]
        oracle = np.array(out["Real_GT_RealGMV"]["weights"])
        long_only = np.array(out["Real_GT_RealGMV_LongOnly"]["weights"])
        self.assertTrue((oracle < 0).any(), "the oracle should short the high-vol asset")
        self.assertFalse(np.allclose(oracle, long_only),
                         "oracle must differ from the long-only run when shorting helps")
        self.assertTrue((long_only >= -1e-12).all())

    def test_one_over_n_is_equal_weights(self):
        out = PortfolioOptimizationEvaluator(
            self.real, self.synth, num_assets_ablation=[6]
        ).evaluate()["n_assets_6"]
        self.assertTrue(np.allclose(out["1N"]["weights"], np.ones(6) / 6))

    def test_every_strategy_weight_sums_to_one(self):
        out = PortfolioOptimizationEvaluator(
            self.real, self.synth, num_assets_ablation=[6]
        ).evaluate()["n_assets_6"]
        for name, res in out.items():
            self.assertAlmostEqual(float(sum(res["weights"])), 1.0, places=8, msg=name)

    def test_turnover_is_none_for_static_strategies(self):
        out = PortfolioOptimizationEvaluator(
            self.real, self.synth, num_assets_ablation=[6]
        ).evaluate()["n_assets_6"]
        for name, res in out.items():
            self.assertIsNone(res["turnover"], msg=name)

    def test_accepts_3d_inputs(self):
        out = PortfolioOptimizationEvaluator(
            self.real[np.newaxis], self.synth[np.newaxis], num_assets_ablation=[6]
        ).evaluate()
        self.assertIn("n_assets_6", out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
