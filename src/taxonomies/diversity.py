"""
Intra-Class Distance (ICD) Metric for Time Series

Computes the average pairwise distance between all time series samples
in a dataset. Supports both Euclidean and DTW (Dynamic Time Warping) metrics.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist
from dtaidistance import dtw


def compute_icd_euclidean(data: np.ndarray) -> float:
    """
    Compute Intra-Class Distance (ICD) for time series using the Euclidean metric.

    Computes the mean pairwise Euclidean distance between samples.
    """
    n_samples, _ = data.shape
    dists = pdist(data, metric='euclidean')
    icd = 2*np.sum(dists) / (n_samples * (n_samples - 1))
    return icd

def compute_icd_dtw(data: np.ndarray) -> float:
    """
    Compute Intra-Class Distance (ICD) for univariate time series using DTW.

    Computes the mean pairwise DTW distance between samples.
    """
    n_samples, _ = data.shape
    sequences = [data[i, :].astype(np.double) for i in range(n_samples)]
    dist_matrix = dtw.distance_matrix(sequences, compact=False)
    upper_sum = np.sum(dist_matrix)
    icd = upper_sum / (n_samples * (n_samples - 1))
    return icd

def calculate_icd(comp_data: np.ndarray, metric: str = "euclidean") -> float:
    """
    Calculate the Intra-Class Distance (ICD) for univariate time series data.

    ICD = (2 / A^2) * sum_{i<j} D(X_i, X_j)
    """
    assert metric in ["euclidean", "dtw"], "Unsupported metric"
    assert comp_data.ndim == 2, "Expected 2D array (n_samples, timesteps)"

    if metric == "euclidean":
        return compute_icd_euclidean(comp_data)
    return compute_icd_dtw(comp_data)
