"""
Metrics for Diversity Taxonomy

This module provides metrics to quantify the diversity of datasets, 
including Euclidean Distance and Dynamic Time Warping (DTW) between original and comparison datasets.
"""

import numpy as np
from scipy.spatial.distance import cdist
from dtaidistance import dtw


def calculate_icd(comp_data, metric="euclidean"):
    """
    Calculate the average pairwise distance (including self-distances) between all points in comp_data.
    Supports 'Euclidean' and 'DTW' (Dynamic Time Warp) metrics.

    Args:
        comp_data (np.ndarray): Array of shape (n_samples, n_features) or (n_samples, sequence_length)
        metric (str): 'Euclidean' or 'DTW'

    Returns:
        float: The sum of all pairwise distances (including self-distances), divided by n_samples^2.
    """
    n_samples = comp_data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if metric == "euclidean":
        distance_matrix = cdist(comp_data, comp_data, metric='euclidean')
    elif metric == "dtw":
        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    dist = dtw.distance(comp_data[i], comp_data[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'Euclidean' or 'DTW'.")
    return np.sum(distance_matrix) / (n_samples ** 2)
