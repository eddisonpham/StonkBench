"""
Intra-Class Distance (ICD) Metric for Multivariate Time Series

This module computes the average pairwise distance between all time series samples
in a dataset. It supports Euclidean and DTW (Dynamic Time Warping) base metrics.
"""
import numpy as np
import torch
from dtaidistance import dtw

from src.utils.conversion_utils import to_numpy_abc


def calculate_icd(comp_data, metric: str = "euclidean") -> float:
    """
    Calculate intra-class distance (ICD) for multivariate time series data, normalized by A^2.

    ICD = (1 / A^2) * sum_{i=1}^A sum_{j=1}^A D(X_i, X_j)

    Args:
        comp_data: Array-like or tensor of shape (A, B, C) or (B, C). The first channel may be a timestamp.
        metric (str): 'euclidean' or 'dtw'

    Returns:
        float: Average distance normalized by A^2.
    """
    metric = metric.lower()
    if metric not in ("euclidean", "dtw"):
        raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'dtw'.")

    data = to_numpy_abc(comp_data)

    A, B, C = data.shape
    distance_sum = 0.0

    for i in range(A):
        for j in range(A):
            dist = 0.0
            if metric == "euclidean":
                for ch in range(C):
                    diff = data[i, :, ch] - data[j, :, ch]
                    dist += np.linalg.norm(diff)
            else:  # dtw
                for ch in range(C):
                    series1 = data[i, :, ch].astype(np.double, copy=False)
                    series2 = data[j, :, ch].astype(np.double, copy=False)
                    dist += dtw.distance_fast(series1, series2)
            distance_sum += dist / C

    icd = distance_sum / (A ** 2)
    return float(icd)
