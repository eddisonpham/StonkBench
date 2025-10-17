"""
Diversity metrics for the taxonomies module.

This module provides a unified interface for diversity evaluation metrics.
"""

# Import actual implementations
from src.evaluation.metrics.diversity import (
    calculate_icd
)

# Re-export for compatibility
__all__ = ['calculate_icd']