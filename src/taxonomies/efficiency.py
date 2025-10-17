"""
Efficiency metrics for the taxonomies module.

This module provides a unified interface for efficiency evaluation metrics.
"""

# Import actual implementations
from src.evaluation.metrics.efficiency import (
    measure_runtime
)

# Re-export for compatibility
__all__ = ['measure_runtime']