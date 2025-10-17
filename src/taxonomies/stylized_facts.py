"""
Stylized facts metrics for the taxonomies module.

This module provides a unified interface for stylized facts evaluation metrics.
"""

# Import actual implementations
from src.evaluation.metrics.stylized_facts import (
    heavy_tails,
    autocorr_raw,
    volatility_clustering,
    long_memory_abs,
    non_stationarity
)

# Re-export for compatibility
__all__ = [
    'heavy_tails',
    'autocorr_raw', 
    'volatility_clustering',
    'long_memory_abs',
    'non_stationarity'
]