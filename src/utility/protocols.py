"""§6 protocols (TSTR, Augmented).

- ``TSTRProtocol`` (§6.1.1): train ``M_r`` on real ``D_train``, train
  ``M̂_g`` on synthetic ``D̂_g``; score both on real ``D_test``. A
  generator is useful iff ``s_g > s_r``; the best generator is
  ``argmax_g s_g``.
- ``AugmentedProtocol`` (§6.1.2): also train ``M̃`` on the full union
  ``D_train ∪ D̂_g`` (no balancing) and score on ``D_test``. The result
  triple is ``{M_r, M̂_g, M̃}``.

The protocols operate *per task* via :class:`BaseUtilityTask`. Tasks
that are ``is_per_channel`` (Options) run C independent training/eval
pairs and aggregate per-channel + per-test-window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch

from src.utility.metrics import MetricToolbox, METRIC_KEYS


# --------------------------------------------------------------------------- #
class BaseProtocol(ABC):
    """Common run helper."""

    @abstractmethod
    def run(
        self,
        task,
        real_train: torch.Tensor,
        real_test: torch.Tensor,
        synthetic_train: torch.Tensor,
        real_train_initial_dollars: torch.Tensor,
        real_test_initial_dollars: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...

    # Helper ---------------------------------------------------------- #
    @staticmethod
    def _aggregate_windows(pnl_per_window: torch.Tensor, task) -> Dict[str, Any]:
        """Given ``(N, L-1)`` per-window PnL, compute U1–U5 mean ± std."""
        windows = []
        for i in range(pnl_per_window.shape[0]):
            windows.append(MetricToolbox.compute(pnl_per_window[i]))
        agg = MetricToolbox.aggregate(windows)
        return agg

    @staticmethod
    def _reshape_init_dollars(
        real_test_initial_dollars: Optional[torch.Tensor], n: int
    ) -> torch.Tensor:
        """Default initial_dollars = $1 for every test window when missing.

        Raises ``ValueError`` if the supplied dollar tensor has the
        wrong size — silently broadcasting the *first* element in such
        cases was a footgun (e.g. ``[1.0, 5.0]`` against ``N=3`` would
        crash mid-tensor multiplication).
        """
        if real_test_initial_dollars is None:
            return torch.ones(n, dtype=torch.float32)
        d = real_test_initial_dollars.float().reshape(-1)
        if d.shape[0] == n:
            return d
        if d.shape[0] == 1:
            return d.expand(n).contiguous()  # scalar → broadcast
        raise ValueError(
            f"initial_dollars has length {d.shape[0]} but there are {n} "
            f"test windows — pass a tensor of shape ({n},), shape (1,) "
            f"for a single shared dollar amount, or None for the $1 default."
        )

    @staticmethod
    def _aggregate_channels(channel_aggregates: List[Dict[str, Any]]) -> Dict[str, Any]:
        return MetricToolbox.aggregate_per_channel(channel_aggregates)


# --------------------------------------------------------------------------- #
class TSTRProtocol(BaseProtocol):
    """§6.1.1 TSTR protocol."""

    name = "tstr"

    def run(
        self,
        task,
        real_train: torch.Tensor,
        real_test: torch.Tensor,
        synthetic_train: torch.Tensor,
        real_train_initial_dollars: torch.Tensor,
        real_test_initial_dollars: torch.Tensor,
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        init_d = self._reshape_init_dollars(real_test_initial_dollars, real_test.shape[0])
        # Train on real.
        m_r_results = self._fit_score(
            task, real_train, real_test, init_d,
            num_epochs=num_epochs, batch_size=batch_size,
            learning_rate=learning_rate, verbose=verbose,
            tag="real",
        )
        # Train on synthetic.
        m_g_results = self._fit_score(
            task, synthetic_train, real_test, init_d,
            num_epochs=num_epochs, batch_size=batch_size,
            learning_rate=learning_rate, verbose=verbose,
            tag="synthetic",
        )

        if task.is_per_channel:
            real_agg = self._aggregate_channels(m_r_results["per_channel"])
            syn_agg = self._aggregate_channels(m_g_results["per_channel"])
            best = self._best_generator(syn_agg, real_agg)
            return {
                "real": m_r_results,
                "synthetic": m_g_results,
                "real_aggregate": real_agg,
                "synthetic_aggregate": syn_agg,
                "useful": best["useful"],
                "score_delta": best["delta"],
            }
        else:
            # Whole-window: ``_fit_score`` returns ``{"aggregate": agg, "extras": ex}``.
            best = self._best_generator(
                m_g_results["aggregate"], m_r_results["aggregate"]
            )
            return {
                "real": m_r_results["aggregate"],
                "synthetic": m_g_results["aggregate"],
                "useful": best["useful"],
                "score_delta": best["delta"],
            }

    # ------------------------------------------------------------------ #
    def _fit_score(
        self, task, train_data, test_data, init_d,
        *, num_epochs, batch_size, learning_rate, verbose, tag: str,
    ) -> Dict[str, Any]:
        if task.is_per_channel:
            num_channels = train_data.shape[-1]
            channel_results = []
            extras_per_channel = []
            for c in range(num_channels):
                if hasattr(task, "run_one_channel"):
                    policy, pnl, ex = task.run_one_channel(
                        train_data[:, :, c],
                        test_data[:, :, c],
                        init_d,
                        num_epochs=num_epochs, batch_size=batch_size,
                        learning_rate=learning_rate, verbose=verbose,
                    )
                else:
                    raise RuntimeError(
                        f"Task {type(task).__name__} marked per-channel "
                        "but doesn't implement run_one_channel()."
                    )
                channel_results.append(self._aggregate_windows(pnl, task))
                extras_per_channel.append(ex)
            return {
                "per_channel": channel_results,
                "extras": extras_per_channel,
            }
        else:
            if not hasattr(task, "run_one"):
                raise RuntimeError(
                    f"Task {type(task).__name__} is whole-window but doesn't "
                    "implement run_one()."
                )
            policy, pnl, ex = task.run_one(
                train_data, test_data, init_d,
                num_epochs=num_epochs, batch_size=batch_size,
                learning_rate=learning_rate, verbose=verbose,
            )
            agg = self._aggregate_windows(pnl, task)
            return {"aggregate": agg, "extras": ex}

    @staticmethod
    def _best_generator(synthetic_results, real_results) -> Dict[str, Any]:
        """Useful iff ``S_synth > S_real``. Higher U1 PnL is better."""
        s_synth = synthetic_results["pnl"]["mean"]
        s_real = real_results["pnl"]["mean"]
        return {"useful": bool(s_synth > s_real), "delta": float(s_synth - s_real)}


# --------------------------------------------------------------------------- #
class AugmentedProtocol(BaseProtocol):
    """§6.1.2 Augmented training (full union, no balancing)."""

    name = "augmented"

    def run(
        self,
        task,
        real_train: torch.Tensor,
        real_test: torch.Tensor,
        synthetic_train: torch.Tensor,
        real_train_initial_dollars: torch.Tensor,
        real_test_initial_dollars: torch.Tensor,
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        init_d = self._reshape_init_dollars(real_test_initial_dollars, real_test.shape[0])
        # Full union along the N axis — keep both data domains intact.
        augmented_train = self._union(real_train, synthetic_train)

        common_kw = dict(num_epochs=num_epochs, batch_size=batch_size,
                         learning_rate=learning_rate, verbose=verbose)
        real_res = TSTRProtocol()._fit_score(
            task, real_train, real_test, init_d, tag="real", **common_kw)
        synth_res = TSTRProtocol()._fit_score(
            task, synthetic_train, real_test, init_d, tag="synthetic", **common_kw)
        aug_res = TSTRProtocol()._fit_score(
            task, augmented_train, real_test, init_d, tag="augmented", **common_kw)

        def _topagg(results: Dict[str, Any]) -> Dict[str, Any]:
            """Collapse per-channel results to top-level U1–U5 mean ± std;
            whole-window tasks already aggregate directly inside ``_fit_score``."""
            if "per_channel" in results:
                return self._aggregate_channels(results["per_channel"])
            return results["aggregate"]

        # Paper §6.1.2 use: "does adding synthetic data to the real
        # training set improve over real-only?" → M̃ vs M_r comparison.
        aug_top = _topagg(aug_res)
        real_top = _topagg(real_res)
        synth_top = _topagg(synth_res)
        aug_best = TSTRProtocol._best_generator(aug_top, real_top)
        tstr_best = TSTRProtocol._best_generator(synth_top, real_top)

        return {
            "real": real_res,
            "synthetic": synth_res,
            "augmented": aug_res,
            "useful": tstr_best["useful"],          # M̂_g vs M_r (TSTR §6.1.1)
            "score_delta": tstr_best["delta"],
            "useful_aug": aug_best["useful"],      # M̃ vs M_r (Augmented §6.1.2)
            "score_delta_aug": aug_best["delta"],
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Concatenate along the window (first) axis — full-union, no
        balancing (§6.1.2)."""
        if a.shape[1:] != b.shape[1:]:
            raise ValueError(
                f"Cannot union: shape mismatch {a.shape} vs {b.shape}"
            )
        return torch.cat([a, b], dim=0)
