"""
Intra-Class Distance (ICD) Metric for Multivariate Time Series

Computes the average pairwise distance between all time series samples
in a dataset. Supports both Euclidean and DTW (Dynamic Time Warping) metrics.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist
from dtaidistance import dtw
import torch


def compute_icd_euclidean(data: np.ndarray) -> np.ndarray:
    """
    Compute Intra-Class Distance (ICD) per channel using Euclidean metric.

    Computes the mean pairwise Euclidean distance between samples, for each channel separately.

    Args:
        data: np.ndarray of shape (n_samples, timesteps, n_channels)

    Returns:
        np.ndarray: Vector of ICD values, one for each channel (shape: n_channels,)
    """
    n_samples, timesteps, n_channels = data.shape
    icd_vals = np.zeros(n_channels, dtype=np.float64)
    for ch in range(n_channels):
        channel_data = data[:, :, ch]
        samples_flat = channel_data.reshape(n_samples, -1)
        dists = pdist(samples_flat, metric='euclidean')
        icd = (2.0 * np.sum(dists)) / (n_samples ** 2)
        icd_vals[ch] = icd
    return icd_vals

def compute_icd_dtw(data: np.ndarray) -> np.ndarray:
    """
    Compute Intra-Class Distance (ICD) per channel using DTW (Dynamic Time Warping).

    Computes the mean pairwise DTW distance between samples, separately for each channel.

    Args:
        data: np.ndarray of shape (n_samples, timesteps, n_channels)

    Returns:
        np.ndarray: Vector of ICD values, one for each channel (shape: n_channels,)
    """
    n_samples, _, n_channels = data.shape
    icd_vals = np.zeros(n_channels, dtype=np.float64)
    for ch in range(n_channels):
        channel_data = [data[i, :, ch].astype(np.double) for i in range(n_samples)]
        channel_dist = dtw.distance_matrix_fast(channel_data, compact=False)
        upper_sum = np.sum(np.triu(channel_dist, k=1))
        icd = (2.0 * upper_sum) / (n_samples ** 2)
        icd_vals[ch] = icd
    return icd_vals

def calculate_icd(comp_data: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Calculate the Intra-Class Distance (ICD) per channel for multivariate time series data.

    ICD[ch] = (2 / A^2) * sum_{i<j} D(X_i^ch, X_j^ch)

    Args:
        comp_data: np.ndarray of shape (n_samples, timesteps, n_channels)
        metric: Distance metric ('euclidean' or 'dtw')

    Returns:
        np.ndarray: Vector of average pairwise distance per channel (shape: n_channels,)
    """
    assert metric in ["euclidean", "dtw"], "Unsupported metric"
    assert comp_data.ndim == 3, "Expected 3D array (n_samples, timesteps, n_channels)"

    if metric == "euclidean":
        return compute_icd_euclidean(comp_data)
    return compute_icd_dtw(comp_data)


