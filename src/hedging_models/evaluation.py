"""Fecamp-style deep hedger evaluator.

Wraps ``FecampHedger`` and ``FecampPortfolioHedger`` for the unified
evaluator. Trains on the *generated* (synthetic) train split and evaluates
PnL on the *real* test split, matching the Fecamp paper protocol.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from src.hedging_models.deep_hedgers.fecamp_hedgers import (
    FecampHedger,
    FecampPortfolioHedger,
)


class FecampDeepHedgerEvaluator:
    """Train a Fecamp hedger on synth data; evaluate on real test split.

    Args:
        synthetic_train / val / test: ``(R, L)`` or ``(R, L, C)`` generated returns.
        real_train / val / test: ``(R, L)`` or ``(R, L, C)`` real data.
        seq_length: window length.
        loss_type: ``cvar`` | ``entropic`` | ``log_utility`` | ``mse``.
        portfolio: True → ``FecampPortfolioHedger`` (multi-asset log-utility).
        hedging_universe: ``"self_only"`` (per-asset only) or ``"basket"``
            (hedge with all available assets — same shape, treated as portfolio).
        transaction_cost: per-unit cost on |delta_change|.
        data_mode: ``synthetic_only`` (default) | ``test_only`` | ``augmented``.
            - ``synthetic_only``: train on ``synthetic_train``; eval on ``real_test``.
            - ``test_only``: train on ``real_train`` only (sanity baseline).
            - ``augmented``: train on ``mix_ratio * synthetic_train``
              + ``(1 - mix_ratio) * real_train`` (NEVER mixes real_test, to keep
              the held-out evaluation target strictly out of training).
        mix_ratio: weight on ``synthetic_train`` in ``augmented`` mode
            (``0.0`` → pure real_train, ``1.0`` → pure synthetic_train).
    """

    _VALID_DATA_MODES = ("synthetic_only", "test_only", "augmented")

    def __init__(
        self,
        synthetic_train: torch.Tensor,
        synthetic_val: torch.Tensor,
        synthetic_test: torch.Tensor,
        real_train: torch.Tensor,
        real_val: torch.Tensor,
        real_test: torch.Tensor,
        seq_length: int,
        num_epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        loss_type: str = "cvar",
        loss_alpha: float = 0.05,
        transaction_cost: float = 0.0,
        hedging_universe: str = "basket",
        portfolio: bool = False,
        n_assets: int = 1,
        data_mode: str = "synthetic_only",
        mix_ratio: float = 0.0,
    ) -> None:
        if data_mode not in self._VALID_DATA_MODES:
            raise ValueError(f"data_mode must be in {self._VALID_DATA_MODES} (got {data_mode!r})")
        if not 0.0 <= mix_ratio <= 1.0:
            raise ValueError(f"mix_ratio must be in [0, 1] (got {mix_ratio})")
        self.synthetic_train = synthetic_train
        self.synthetic_val = synthetic_val
        self.synthetic_test = synthetic_test
        self.real_train = real_train
        self.real_val = real_val
        self.real_test = real_test
        self.seq_length = int(seq_length)
        self.num_epochs = int(num_epochs)
        self.batch_size = max(1, int(batch_size))
        self.learning_rate = float(learning_rate)
        self.loss_type = str(loss_type)
        self.loss_alpha = float(loss_alpha)
        self.transaction_cost = float(transaction_cost)
        self.hedging_universe = str(hedging_universe)
        self.portfolio = bool(portfolio)
        self.n_assets = int(n_assets)
        self.data_mode = str(data_mode)
        self.mix_ratio = float(mix_ratio)
        self.hedger: Optional[torch.nn.Module] = None
        self.results: Dict[str, Any] = {}
        # One-shot warning guards for augmented-mode subsampling messages so the
        # per-(mode, mix, channel) loop doesn't spam logs in print-loop disease.
        self._warned_synth_short: bool = False
        self._warned_real_short: bool = False

    def _flatten_for_nn(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            # Pass-through; FecampHedger.forward internally flattens (N, L, C) -> (N, L*C).
            # NOTE: an earlier implementation reshaped (N, L, C) -> (N, seq_length, C) here,
            # which silently corrupted any input whose window length L != self.seq_length.
            return x
        if x.ndim == 2:
            return x.unsqueeze(-1)
        raise ValueError(f"Unexpected shape {tuple(x.shape)} for hedger input")

    def _build_training_tensor(self) -> torch.Tensor:
        """Return the ``(N, L)`` or ``(N, L, C)`` training tensor.

        The augmented mode mixes real_train (NEVER real_test) and synthetic_train.
        Reference size n is the synthetic_train length; ``n_synth``, ``n_real``
        floors so an over-budget real slice falls back to the available rows.
        Two warnings are emitted at most once per evaluator instance to keep
        per-mode × per-mix × per-channel log spam bounded (5 modes × 5 mixes ×
        50 channels would otherwise print 1,250 lines).
        """
        if self.data_mode == "synthetic_only":
            return self.synthetic_train
        if self.data_mode == "test_only":
            return self.real_train
        # augmented
        n = max(len(self.synthetic_train), 1)
        n_synth = min(int(n * self.mix_ratio), len(self.synthetic_train))
        n_real = min(int(n * (1.0 - self.mix_ratio)), len(self.real_train))
        if n_synth == 0 and n_real == 0:
            return self.synthetic_train[:0]
        synth_part = self.synthetic_train[:n_synth] if n_synth > 0 else self.synthetic_train[:0]
        # One-shot guard so per-channel loops don't spam a (mode, mix, ch) grid
        # of identical messages.
        if n_synth < int(n * self.mix_ratio) and not self._warned_synth_short:
            print(
                f"[WARN] FecampDeepHedgerEvaluator: mix_ratio={self.mix_ratio} requested "
                f"{int(n * self.mix_ratio)} synth rows but only {n_synth} available; "
                f"subsampled to available data."
            )
            self._warned_synth_short = True
        if n_real < int(n * (1.0 - self.mix_ratio)) and not self._warned_real_short:
            print(
                f"[WARN] FecampDeepHedgerEvaluator: "
                f"{int(n * (1.0 - self.mix_ratio))} real_train rows requested but only "
                f"{n_real} available; subsampled (real_test is never used)."
            )
            self._warned_real_short = True
        if synth_part.shape[0] == 0:
            return self.real_train[:n_real]
        if self.real_train[:n_real].shape[0] == 0:
            return synth_part
        return torch.cat([synth_part, self.real_train[:n_real]], dim=0)

    def evaluate(self) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.portfolio:
            self.hedger = FecampPortfolioHedger(
                seq_length=self.seq_length,
                n_assets=self.n_assets,
                loss_type=self.loss_type,
                transaction_cost=self.transaction_cost,
            ).to(device)
        else:
            self.hedger = FecampHedger(
                seq_length=self.seq_length,
                loss_type=self.loss_type,
                loss_alpha=self.loss_alpha,
            ).to(device)

        train_tensor = self._build_training_tensor()
        train_in = self._flatten_for_nn(train_tensor).to(device)
        optimizer = torch.optim.Adam(self.hedger.parameters(), lr=self.learning_rate)

        n = train_in.shape[0]
        train_loss_history: list[float] = []
        for _epoch in range(self.num_epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                batch = train_in[idx]
                optimizer.zero_grad()
                loss = self.hedger.compute_loss(batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            train_loss_history.append(epoch_loss / max(n_batches, 1))

        # Evaluate on the REAL test split (synth model should generalize here).
        test_in = self._flatten_for_nn(self.real_test).to(device)
        self.hedger.eval()
        with torch.no_grad():
            deltas = self.hedger(test_in)
            pnl = self.hedger.compute_pnl(test_in, deltas, transaction_cost=self.transaction_cost)
        pnl_np = pnl.detach().cpu().numpy()

        results: Dict[str, Any] = {
            "loss_type": self.loss_type,
            "loss_alpha": self.loss_alpha,
            "hedging_universe": self.hedging_universe,
            "transaction_cost": self.transaction_cost,
            "data_mode": self.data_mode,
            "mix_ratio": self.mix_ratio,
            "num_train_samples": int(n),
            "num_test_samples": int(test_in.shape[0]),
            "train_loss_history": train_loss_history,
            "pnl_mean": float(pnl_np.mean()),
            "pnl_std": float(pnl_np.std(ddof=0)),
            "pnl_min": float(pnl_np.min()),
            "pnl_max": float(pnl_np.max()),
            "pnl_q05": float(np.percentile(pnl_np, 5)),
            "pnl_q50": float(np.percentile(pnl_np, 50)),
            "pnl_q95": float(np.percentile(pnl_np, 95)),
        }
        var_5 = float(np.percentile(pnl_np, 5))
        tail = pnl_np[pnl_np <= var_5]
        if tail.size > 0:
            results["cvar_5pct"] = float(tail.mean())
        self.results = results
        return results
