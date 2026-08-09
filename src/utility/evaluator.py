"""Orchestrator for the §6 utility pipeline.

Given a :class:`BaseUtilityTask` and the data partitions (real train /
real test / synthetic train / initial dollars), runs the chosen
:mod:`protocols` and aggregates the U1–U5 mean ± std report.

Typical usage:

>>> task = OptionsTask(seq_length=252)
>>> ev = UtilityEvaluator(task, protocol="tstr")
>>> ev.run(real_train, real_test, synthetic_train, dollars)

Returns a nested dict keyed by ``{"protocol": {"real": ..., "synthetic": ...}}``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.utility.protocols import TSTRProtocol, AugmentedProtocol


class UtilityEvaluator:
    """High-level orchestrator. Plugs a task + protocol + data together."""

    def __init__(
        self,
        task,
        protocol: str = "tstr",
        *,
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        verbose: bool = False,
    ):
        if protocol not in ("tstr", "augmented"):
            raise ValueError(f"protocol must be tstr|augmented, got {protocol!r}")
        self.task = task
        self.protocol_name = protocol
        self.protocol = TSTRProtocol() if protocol == "tstr" else AugmentedProtocol()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    def run(
        self,
        real_train: torch.Tensor,
        real_test: torch.Tensor,
        synthetic_train: torch.Tensor,
        real_train_initial_dollars: Optional[torch.Tensor] = None,
        real_test_initial_dollars: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if real_train_initial_dollars is None:
            real_train_initial_dollars = torch.ones(real_train.shape[0])
        if real_test_initial_dollars is None:
            real_test_initial_dollars = torch.ones(real_test.shape[0])

        out = self.protocol.run(
            self.task,
            real_train=real_train,
            real_test=real_test,
            synthetic_train=synthetic_train,
            real_train_initial_dollars=real_train_initial_dollars,
            real_test_initial_dollars=real_test_initial_dollars,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
        )
        out["task"] = self.task.task_name
        out["protocol"] = self.protocol_name
        return out

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.task_name,
            "protocol": self.protocol_name,
            "is_per_channel": self.task.is_per_channel,
            "pnl_returns": self.task.pnl_returns,
        }
