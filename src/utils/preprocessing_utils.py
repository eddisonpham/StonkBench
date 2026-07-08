"""Minimal preprocessing helper shared by utility metrics."""

from __future__ import annotations

from typing import Tuple

import torch


class LogReturnTransformation:
    """Transform prices to log returns and invert them back to prices."""

    def transform(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_returns = torch.log(data[1:] / data[:-1])
        return log_returns, data[0]

    def inverse_transform(self, log_returns: torch.Tensor, initial_value: torch.Tensor) -> torch.Tensor:
        prices = [initial_value]
        for r in log_returns:
            prices.append(prices[-1] * torch.exp(r))
        return torch.stack(prices)