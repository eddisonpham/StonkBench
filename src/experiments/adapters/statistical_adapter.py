"""Statistical model adapters — multivariate-only (post 2026-07-28 cleanup).

The 5 canonical statistical adapters in this file all produce (R, L, C) output with
genuine cross-channel correlation (no longer the per-channel-wrapped univariate OO
shim from before). GBM and OU have been removed entirely; their roles were subsumed
into the multivariate Merton/DEJD/GARCH stack. Stationary Block Bootstrap is a new
sibling to Moving Block Bootstrap preserving cross-channel structure.

| Adapter                          | Vendor model              | Multivariate strategy                                  |
|----------------------------------|---------------------------|-------------------------------------------------------|
| StatisticalMertonAdapter         | MertonJumpDiffusion       | Per-channel params + Cholesky-of-cov diffusion       |
| StatisticalDEJDAdapter           | DoubleExponentialJumpDiff | Per-channel params + Cholesky-of-cov diffusion       |
| StatisticalDCCGARCHAdapter     | DCC-GARCH(1,1)            | Engle (2002): GARCH vol + time-varying DCC correlation |
| BlockBootstrapAdapter            | BlockBootstrap (MBB)      | Multivariate block resample (Kunsch 1989)             |
| StationaryBlockBootstrapAdapter  | StationaryBlockBootstrap  | Multivariate Geometric-length block resample (Politis & Romano 1994) |
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.models.statistical.block_bootstrap import BlockBootstrap
from src.models.statistical.de_jump_diffusion import DoubleExponentialJumpDiffusion
from src.models.statistical.dcc_garch import DCCGARCH
from src.models.statistical.merton_jump_diffusion import MertonJumpDiffusion
from src.models.statistical.stationary_block_bootstrap import StationaryBlockBootstrap


# ---------------------------------------------------------------------------
# Multivariate Merton jump-diffusion.
# ---------------------------------------------------------------------------


class StatisticalMertonAdapter(ModelAdapter):
    """Multivariate Merton jump-diffusion (Cholesky-of-cov diffusion + per-channel jumps)."""

    model_name = "MertonAdapter"
    supports_arbitrary_generation = True

    def __init__(self) -> None:
        super().__init__()
        self.model: MertonJumpDiffusion | None = None
        self.num_channels = 0

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        train = fit_input.batch.train
        if train.ndim == 1:
            train = train.unsqueeze(-1)
        if train.ndim != 2:
            raise ValueError("StatisticalMertonAdapter expects train split shaped (L, C)")
        self.num_channels = train.shape[1]
        self.model = MertonJumpDiffusion()
        self.model.fit(train)
        self._is_fitted = True
        return {"num_channels": self.num_channels, "multivariate": True}

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")
        data = self.model.generate(num_samples, generation_length, seed)
        if data.ndim == 2:
            data = data.unsqueeze(-1)
        return AdapterGenerateOutput(
            data=data.float(),
            logs={"generator": "multivariate_merton_cholesky_jumps"},
            extra_metadata={"num_channels": self.num_channels, "multivariate": True},
        )


# ---------------------------------------------------------------------------
# Multivariate Double-Exponential Jump-Diffusion.
# ---------------------------------------------------------------------------


class StatisticalDEJDAdapter(ModelAdapter):
    """Multivariate DEJD (Kou 2002) — Cholesky-of-cov diffusion + per-channel jumps."""

    model_name = "DEJDAdapter"
    supports_arbitrary_generation = True

    def __init__(self) -> None:
        super().__init__()
        self.model: DoubleExponentialJumpDiffusion | None = None
        self.num_channels = 0

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        train = fit_input.batch.train
        if train.ndim == 1:
            train = train.unsqueeze(-1)
        if train.ndim != 2:
            raise ValueError("StatisticalDEJDAdapter expects train split shaped (L, C)")
        self.num_channels = train.shape[1]
        self.model = DoubleExponentialJumpDiffusion()
        self.model.fit(train)
        self._is_fitted = True
        return {"num_channels": self.num_channels, "multivariate": True}

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")
        data = self.model.generate(num_samples, generation_length, seed)
        if data.ndim == 2:
            data = data.unsqueeze(-1)
        return AdapterGenerateOutput(
            data=data.float(),
            logs={"generator": "multivariate_dejd_cholesky_jumps"},
            extra_metadata={"num_channels": self.num_channels, "multivariate": True},
        )


# ---------------------------------------------------------------------------
# Multivariate GARCH(1,1) with frozen-correlation (Cholesky-of-residual-corr).
# ---------------------------------------------------------------------------


class StatisticalDCCGARCHAdapter(ModelAdapter):
    """Multivariate DCC-GARCH(1,1) — Engle (2002): per-channel GARCH vol + time-varying DCC correlation.

    Natively multivariate: the cross-channel dependence is carried by the Q_t / R_t
    matrices that evolve jointly via the DCC recursion, not by a static Cholesky
    overlay on per-channel draws.
    """

    model_name = "DCCGARCHAdapter"
    supports_arbitrary_generation = True

    def __init__(self) -> None:
        super().__init__()
        self.model: DCCGARCH | None = None
        self.num_channels = 0

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        train = fit_input.batch.train
        if train.ndim == 1:
            train = train.unsqueeze(-1)
        if train.ndim != 2:
            raise ValueError("StatisticalDCCGARCHAdapter expects train split shaped (L, C)")
        self.num_channels = train.shape[1]
        self.model = DCCGARCH()
        self.model.fit(train)
        self._is_fitted = True
        return {
            "num_channels": self.num_channels,
            "multivariate": True,
            "dcc_alpha": self.model.alpha_dcc,
            "dcc_beta": self.model.beta_dcc,
        }

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")
        data = self.model.generate(num_samples, generation_length, seed)
        if data.ndim == 2:
            data = data.unsqueeze(-1)
        return AdapterGenerateOutput(
            data=data.float(),
            logs={"generator": "dcc_garch_engle2002"},
            extra_metadata={
                "num_channels": self.num_channels,
                "multivariate": True,
                "dcc_alpha": self.model.alpha_dcc,
                "dcc_beta": self.model.beta_dcc,
            },
        )


# ---------------------------------------------------------------------------
# Multivariate Moving Block Bootstrap (Kunsch 1989 / Politis & Romano 1992).
# ---------------------------------------------------------------------------


class BlockBootstrapAdapter(ModelAdapter):
    """Multivariate Moving Block Bootstrap — fixed block size, contiguous resamples."""

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
        return {
            "num_channels": self.num_channels,
            "block_size": self.block_size,
            "multivariate": True,
        }

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")
        data = self.model.generate(num_samples, generation_length, seed)
        if data.ndim == 2:
            data = data.unsqueeze(-1)
        return AdapterGenerateOutput(
            data=data.float(),
            logs={"generator": "aligned_block_bootstrap_mbb"},
            extra_metadata={
                "num_channels": self.num_channels,
                "block_size": self.block_size,
                "multivariate": True,
            },
        )


# ---------------------------------------------------------------------------
# Multivariate Stationary Block Bootstrap (Politis & Romano 1994).
# ---------------------------------------------------------------------------


class StationaryBlockBootstrapAdapter(ModelAdapter):
    """Multivariate Stationary Block Bootstrap — Geometric-distributed block lengths."""

    model_name = "StationaryBlockBootstrapAdapter"
    supports_arbitrary_generation = True

    def __init__(self, expected_block_length: float = 32.0) -> None:
        super().__init__()
        self.expected_block_length = expected_block_length
        self.model: StationaryBlockBootstrap | None = None
        self.num_channels = 0

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        train = fit_input.batch.train
        if train.ndim == 1:
            train = train.unsqueeze(-1)
        if train.ndim != 2:
            raise ValueError("StationaryBlockBootstrapAdapter expects train split shaped (L, C)")
        self.num_channels = train.shape[1]
        self.model = StationaryBlockBootstrap(expected_block_length=self.expected_block_length)
        self.model.fit(train)
        self._is_fitted = True
        return {
            "num_channels": self.num_channels,
            "expected_block_length": self.expected_block_length,
            "multivariate": True,
        }

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")
        data = self.model.generate(num_samples, generation_length, seed)
        if data.ndim == 2:
            data = data.unsqueeze(-1)
        return AdapterGenerateOutput(
            data=data.float(),
            logs={"generator": "stationary_block_bootstrap_sbb"},
            extra_metadata={
                "num_channels": self.num_channels,
                "expected_block_length": self.expected_block_length,
                "multivariate": True,
            },
        )
