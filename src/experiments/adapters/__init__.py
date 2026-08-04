"""Statistical + DL adapter re-exports.

Post 2026-07-28 cleanup: gbm_adapter and ou_process have been removed; stationary_block_bootstrap
has been added. All statistical adapters in this list are multivariate by construction (Cholesky-of-cov
diffusion for Merton/DEJD/GARCH; full block resampling for MBB and SBB).
"""
from src.experiments.adapters.statistical_adapter import (
    BlockBootstrapAdapter,
    StationaryBlockBootstrapAdapter,
    StatisticalDEJDAdapter,
    StatisticalGARCH11Adapter,
    StatisticalMertonAdapter,
)

__all__ = [
    "BlockBootstrapAdapter",
    "StationaryBlockBootstrapAdapter",
    "StatisticalMertonAdapter",
    "StatisticalDEJDAdapter",
    "StatisticalGARCH11Adapter",
]
