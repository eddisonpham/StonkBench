"""
Utility functions for plotting: colors, formatting, and data manipulation.
"""

import numpy as np
import seaborn as sns
import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from matplotlib.colors import LinearSegmentedColormap

# Color palettes
def get_white_to_green_palette(n_colors: int) -> List[Tuple[float, float, float]]:
    return [(1 - t, 1, 1 - t) for t in np.linspace(0, 1, n_colors)]


def get_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    """
    Generate n visually distinct colors.

    Args:
        n: Number of colors.

    Returns:
        List of RGB color tuples.
    """
    pal = sns.color_palette("tab10", n)
    return [tuple(c) for c in pal]


# Default color palettes
DEFAULT_MODEL_COLORS = get_white_to_green_palette(8)
DEFAULT_LINE_COLORS = get_distinct_colors(8)


# Formatting utilities
def scientific_fmt(x: Any, precision: int = 2) -> str:
    """
    Format a number with scientific notation based on magnitude.

    Args:
        x: Value to format (float/int/str).
        precision: Number of decimals.

    Returns:
        Formatted string.
    """
    try:
        x = float(x)
    except (ValueError, TypeError):
        return str(x)
    
    if np.isnan(x):
        return "nan"
    if x == 0:
        return f"{x:.{precision}f}"
    
    absx = abs(x)
    if absx < 1e-2 or absx > 1e4:
        return f"{x:.{precision}e}"
    return f"{x:.{precision}f}"


def format_minmax_label(min_val: float, max_val: float, precision: int = 3) -> str:
    """
    Return a string representation of min and max values in interval notation.

    Args:
        min_val: Minimum value.
        max_val: Maximum value.
        precision: Number of decimal places.

    Returns:
        Interval string or empty string if values are nan.
    """
    if np.isnan(min_val) or np.isnan(max_val):
        return ""
    return f"[{scientific_fmt(min_val, precision)}, {scientific_fmt(max_val, precision)}]"


# Data manipulation utilities
def make_blank_column(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Create an array of NaNs.

    Args:
        shape: Shape of the array.

    Returns:
        Array filled with np.nan.
    """
    return np.full(shape, np.nan)


def extract_metric_value(model_data: Dict[str, Any], metric_name: str) -> float:
    """
    Extract a metric value from model data.

    Args:
        model_data: Dictionary with model metrics.
        metric_name: Name of the metric.

    Returns:
        Metric value or np.nan if unavailable.
    """
    if metric_name in model_data:
        value = model_data[metric_name]
        if isinstance(value, dict):
            return float(value.get('diff', np.nan))
        if isinstance(value, (float, int)):
            return float(value)
        return np.nan
    
    # Check nested in utility
    if metric_name == 'spearman_correlation':
        utility = model_data.get('utility', {})
        algo_comp = utility.get('algorithm_comparison', {})
        return float(algo_comp.get('spearman_correlation', np.nan))
    
    if metric_name == 'spearman_correlation_mixed':
        utility = model_data.get('utility', {})
        algo_comp = utility.get('algorithm_comparison', {})
        return float(algo_comp.get('spearman_correlation_mixed', np.nan))
    
    return np.nan


# Colormap utilities
def get_white_green_colormap() -> LinearSegmentedColormap:
    """Get a white-to-green colormap for heatmaps."""
    return LinearSegmentedColormap.from_list("WhiteGreen", ["white", "green"])


# Plotting constants
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (10, 6)
HEATMAP_ANNOTATION_FMT = ".2f"
MINMAX_FONTSIZE = 8


# Figure loading utilities
def load_sequence_data(results_dir: Path, seq_len: int) -> Dict[str, Any]:
    """
    Load metrics data for the specified sequence length.

    Args:
        results_dir: Base results directory.
        seq_len: Sequence length.

    Returns:
        Dictionary mapping model names to their metrics data.
    """
    seq_folder = results_dir / f"seq_{seq_len}"
    if not seq_folder.exists():
        return {}
    
    data = {}
    for model_folder in seq_folder.iterdir():
        if model_folder.is_dir():
            metrics_file = model_folder / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data[model_folder.name] = json.load(f)
    
    return data


def load_ablation_data(results_dir: Path, sequence_lengths: Optional[List[int]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load metrics data for ablation study sequence lengths.

    Args:
        results_dir: Base results directory.
        sequence_lengths: List of sequence lengths to load. If None, uses default [60, 120, 180, 240, 300].

    Returns:
        Dictionary with structure: {seq_name: {model_name: metrics_dict}}.
    """
    if sequence_lengths is None:
        sequence_lengths = [60, 120, 180, 240, 300]
    
    ablation_data = {}
    for seq_len in sequence_lengths:
        seq_key = f"seq_{seq_len}"
        seq_data = load_sequence_data(results_dir, seq_len)
        if seq_data:
            ablation_data[seq_key] = seq_data
    
    return ablation_data


def extract_sequence_lengths(ablation_data: Dict[str, Dict[str, Any]]) -> List[int]:
    """
    Extract sorted sequence lengths from ablation data keys.

    Args:
        ablation_data: Ablation data dictionary.

    Returns:
        Sorted list of sequence lengths.
    """
    seq_lengths = []
    for key in ablation_data.keys():
        match = re.match(r'seq_(\d+)', key)
        if match:
            seq_lengths.append(int(match.group(1)))
    
    return sorted(seq_lengths)


def get_models_from_data(data: Dict[str, Any]) -> List[str]:
    """
    Extract model names from data dictionary.

    Args:
        data: Data dictionary (either main_data or first entry of ablation_data).

    Returns:
        List of model names.
    """
    if not data:
        return []
    
    # If data is nested (ablation_data structure)
    if isinstance(next(iter(data.values())), dict) and any(
        'utility' in v or 'mdd' in v for v in data.values() if isinstance(v, dict)
    ):
        return list(data.keys())
    
    # If data is flat (main_data structure)
    return list(data.keys())

