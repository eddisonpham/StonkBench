from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Type

import torch

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.models.statistical.block_bootstrap import BlockBootstrap
from src.models.statistical.de_jump_diffusion import DoubleExponentialJumpDiffusion
from src.models.statistical.garch11 import GARCH11
from src.models.statistical.gbm import GeometricBrownianMotion
from src.models.statistical.merton_jump_diffusion import MertonJumpDiffusion
from src.models.statistical.ou_process import OUProcess


class StatisticalGBMAdapter(ModelAdapter):
    """
    Multivariate-compatible adapter using independent GBM per channel.

    Preserves the univariate statistical model I/O contract while standardizing
    benchmark output to (R, L, C).
    """

    model_name = "GBMAdapter"
    supports_arbitrary_generation = True

    def __init__(self) -> None:
        super().__init__()
        self.models: List[GeometricBrownianMotion] = []
        self.num_channels = 0

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        train = fit_input.batch.train
        if train.ndim == 1:
            train = train.unsqueeze(-1)
        if train.ndim != 2:
            raise ValueError("StatisticalGBMAdapter expects train split shaped (L, C)")

        self.num_channels = train.shape[1]
        self.models = []
        for c in range(self.num_channels):
            model = GeometricBrownianMotion()
            model.fit(train[:, c])
            self.models.append(model)

        self._is_fitted = True
        return {"num_channels": self.num_channels}

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before generate().")

        channels = []
        for c, model in enumerate(self.models):
            channels.append(model.generate(num_samples, generation_length, seed + c))
        data = torch.stack(channels, dim=-1)  # (R, L, C)
        return AdapterGenerateOutput(
            data=data.float(),
            logs={"generator": "independent_gbm_per_channel"},
            extra_metadata={"num_channels": self.num_channels},
        )


class _IndependentStatisticalAdapter(ModelAdapter):
    """
    Generic adapter for univariate statistical models applied independently per channel.
    """

    model_cls: Type = GeometricBrownianMotion
    generator_label = "independent_per_channel"
    model_name = "StatisticalAdapter"
    supports_arbitrary_generation = True

    def __init__(self) -> None:
        super().__init__()
        self.models: List = []
        self.num_channels = 0

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        train = fit_input.batch.train
        if train.ndim == 1:
            train = train.unsqueeze(-1)
        if train.ndim != 2:
            raise ValueError(f"{self.__class__.__name__} expects train split shaped (L, C)")

        self.num_channels = train.shape[1]
        self.models = []
        for c in range(self.num_channels):
            model = self.model_cls()
            model.fit(train[:, c])
            self.models.append(model)

        self._is_fitted = True
        return {"num_channels": self.num_channels}

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before generate().")

        channels = []
        for c, model in enumerate(self.models):
            channel = model.generate(num_samples, generation_length, seed + c)
            if channel.ndim == 3:
                channel = channel.squeeze(-1)
            channels.append(channel)
        data = torch.stack(channels, dim=-1)
        return AdapterGenerateOutput(
            data=data.float(),
            logs={"generator": self.generator_label},
            extra_metadata={"num_channels": self.num_channels},
        )


class StatisticalOUAdapter(_IndependentStatisticalAdapter):
    model_cls = OUProcess
    generator_label = "independent_ou_per_channel"
    model_name = "OUAdapter"


class StatisticalMertonAdapter(_IndependentStatisticalAdapter):
    model_cls = MertonJumpDiffusion
    generator_label = "independent_merton_per_channel"
    model_name = "MertonAdapter"


class StatisticalDEJDAdapter(_IndependentStatisticalAdapter):
    model_cls = DoubleExponentialJumpDiffusion
    generator_label = "independent_dejd_per_channel"
    model_name = "DEJDAdapter"


class StatisticalGARCH11Adapter(_IndependentStatisticalAdapter):
    model_cls = GARCH11
    generator_label = "independent_garch11_per_channel"
    model_name = "GARCH11Adapter"


class BlockBootstrapAdapter(ModelAdapter):
    """
    Multivariate block bootstrap adapter.

    Resamples contiguous blocks from the full train series, preserving cross-channel
    dependence within each block.
    """

    model_name = "BlockBootstrapAdapter"
    supports_arbitrary_generation = True

    def __init__(self, block_size: int = 32) -> None:
        super().__init__()
        self.block_size = block_size
        self.model: BlockBootstrap | None = None
        self.num_channels = 0

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        train = fit_input.batch.train
        if train.ndim == 1:
            train = train.unsqueeze(-1)
        if train.ndim != 2:
            raise ValueError("BlockBootstrapAdapter expects train split shaped (L, C)")

        self.num_channels = train.shape[1]
        self.model = BlockBootstrap(block_size=self.block_size)
        self.model.fit(train)
        self._is_fitted = True
        return {"num_channels": self.num_channels, "block_size": self.block_size}

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")

        data = self.model.generate(num_samples, generation_length, seed)
        if data.ndim == 2:
            data = data.unsqueeze(-1)
        return AdapterGenerateOutput(
            data=data.float(),
            logs={"generator": "aligned_block_bootstrap"},
            extra_metadata={"num_channels": self.num_channels, "block_size": self.block_size},
        )
