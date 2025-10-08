"""
Intra-Class Distance (ICD) Metric for Multivariate Time Series

This module computes the average pairwise distance between all
time series samples in a dataset. It supports Euclidean and DTW (Dynamic Time Warping)
metrics, and correctly handles multi-channel (multivariate) data.

For multi-channel data, the distance between two samples is defined as
the average of the per-channel distances.
"""
import numpy as np
from dtaidistance import dtw


def calculate_icd(comp_data: np.ndarray, metric: str = "euclidean") -> float:
    """
    Calculate intra-class distance (ICD) for multivariate time series data,
    normalized by the dataset size squared (R^2).

    ICD = (1 / R^2) * sum_{i=1}^R sum_{j=1}^R D(X_i, X_j)

    Args:
        comp_data (np.ndarray): Array of shape (R, L, N)
            R: number of samples
            L: length of each time series
            N: number of channels
        metric (str): Distance metric to use ('euclidean' or 'dtw')

    Returns:
        float: Average distance normalized by R^2.
    """
    metric = metric.lower()
    if metric not in ("euclidean", "dtw"):
        raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'dtw'.")

    R, L, N = comp_data.shape
    distance_sum = 0.0

    for i in range(R):
        for j in range(R):
            dist = 0.0
            if metric == "euclidean":
                for ch in range(N):
                    diff = comp_data[i, :, ch] - comp_data[j, :, ch]
                    dist += np.linalg.norm(diff)
            elif metric == "dtw":
                for ch in range(N):
                    dist += dtw.distance_fast(comp_data[i, :, ch], comp_data[j, :, ch])
            distance_sum += dist / N

    icd = distance_sum / (R ** 2)
    return icd
