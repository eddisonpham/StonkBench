"""
Intra-Class Distance (ICD) Metric for Multivariate Time Series

This module computes the average pairwise distance between all time series samples
in a dataset. It supports Euclidean and DTW (Dynamic Time Warping) base metrics.
"""
import numpy as np
import torch
from dtaidistance import dtw

from src.utils.conversion_utils import to_numpy_abc


def euclidean_distance(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two multichannel time series samples.

    Args:
        sample1, sample2: Arrays of shape (timesteps, channels)
    """
    assert sample1.shape == sample2.shape, "Error:Samples must have same shape"
    timesteps, channels = sample1.shape
    dist = 0.0
    for ch in range(channels):
        diff = sample1[:, ch] - sample2[:, ch]
        dist += np.linalg.norm(diff)
    return dist / channels

def dtw_distance(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Compute DTW distance between two multichannel time series samples using dtaidistance.dtw.
    """
    assert sample1.shape == sample2.shape, "Samples must have same shape"
    timesteps, channels = sample1.shape
    dist = 0.0
    for ch in range(channels):
        s1 = sample1[:, ch].astype(np.double, copy=False)
        s2 = sample2[:, ch].astype(np.double, copy=False)
        d = dtw.distance(s1, s2)
        dist += d
    return dist / channels

def calculate_icd(comp_data, metric: str = "euclidean") -> float:
    """
    Calculate intra-class distance (ICD) for multivariate time series data.

    ICD = (1 / A^2) * sum_{i=1}^A sum_{j=1}^A D(X_i, X_j)

    Optimization:
        Only compute upper triangle (i < j) since D(i,j) == D(j,i) and D(i,i) = 0.
    """
    data = to_numpy_abc(comp_data)
    A, B, C = data.shape
    distance_sum = 0.0

    if metric == "euclidean":
        distance_func = euclidean_distance
    elif metric == "dtw":
        distance_func = dtw_distance

    # Upper triangle optimization
    for i in range(A):
        for j in range(i + 1, A):
            dist = distance_func(data[i], data[j])
            distance_sum += dist

    # Multiply by 2 (symmetry) and normalize by A^2
    icd = (2 * distance_sum) / (A ** 2)
    return float(icd)

