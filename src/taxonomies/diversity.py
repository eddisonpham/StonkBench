"""
Intra-Class Distance (ICD) Metric for Multivariate Time Series

Computes the average pairwise distance between all time series samples
in a dataset. Supports both Euclidean and DTW (Dynamic Time Warping) metrics.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist
from dtaidistance import dtw
from src.utils.conversion_utils import to_numpy_abc
import torch


def _compute_icd_euclidean(data: np.ndarray) -> float:
    """
    Compute Intra-Class Distance (ICD) using Euclidean metric.

    Uses scipy.spatial.distance.pdist for efficient pairwise computation
    (C-optimized, memory-efficient).

    Args:
        data: np.ndarray of shape (n_samples, timesteps, n_channels)

    Returns:
        float: Mean pairwise Euclidean distance between samples.
    """
    n_samples, timesteps, n_channels = data.shape
    flattened = data.reshape(n_samples, -1)
    dists = pdist(flattened, metric='euclidean')
    icd = (2.0 * np.sum(dists)) / (n_samples ** 2)
    return float(icd)


def _compute_icd_dtw(data: np.ndarray) -> float:
    """
    Compute Intra-Class Distance (ICD) using DTW (Dynamic Time Warping).

    Computes DTW per channel and averages across channels.

    Args:
        data: np.ndarray of shape (n_samples, timesteps, n_channels)

    Returns:
        float: Mean pairwise DTW distance between samples.
    """
    n_samples, _, n_channels = data.shape
    dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)

    for ch in range(n_channels):
        channel_data = [data[i, :, ch].astype(np.double) for i in range(n_samples)]
        channel_dist = dtw.distance_matrix_fast(channel_data, compact=False)
        dist_matrix += channel_dist

    dist_matrix /= n_channels
    upper_sum = np.sum(np.triu(dist_matrix, k=1))
    icd = (2.0 * upper_sum) / (n_samples ** 2)
    return float(icd)

def calculate_icd(comp_data: np.ndarray | torch.Tensor, metric: str = "euclidean") -> float:
    """
    Calculate the Intra-Class Distance (ICD) for multivariate time series data.

    ICD = (2 / A^2) * sum_{i<j} D(X_i, X_j)

    Args:
        comp_data: np.ndarray or torch.Tensor of shape (n_samples, timesteps, n_channels)
        metric: Distance metric ('euclidean' or 'dtw')

    Returns:
        float: Average pairwise distance across all samples.
    """
    assert metric in ["euclidean", "dtw"], "Unsupported metric"
    assert comp_data.ndim == 3, "Expected 3D array (n_samples, timesteps, n_channels)"

    data = to_numpy_abc(comp_data)
    if metric == "euclidean":
        return _compute_icd_euclidean(data)
    elif metric == "dtw":
        return _compute_icd_dtw(data)

