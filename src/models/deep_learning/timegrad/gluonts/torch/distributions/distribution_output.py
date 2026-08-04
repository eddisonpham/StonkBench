import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class DistributionOutput(ABC):
    args_dim: dict = {}
    dim: int = 0

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        ...

    @classmethod
    def domain_map(cls, *args, **kwargs):
        return ()

    def distribution(self, distr_args, scale=None):
        return None

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)

    def get_args_proj(self, in_features):
        return _IdentityProj(in_features)


class _IdentityProj(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.proj = nn.Linear(in_features, in_features)

    def forward(self, x):
        return (self.proj(x),)
