"""
Stylized Facts Metrics for Financial Time Series

This module implements quantitative metrics for evaluating the presence of stylized facts
in financial time series data. Stylized facts are statistical properties commonly observed
in real-world financial markets, such as:

- Heavy tails (excess kurtosis) in returns
- Absence of autocorrelation in raw returns
- Volatility clustering (autocorrelation in squared returns)
- Long memory in absolute returns

The provided functions operate on multivariate time series data (e.g., Open, High, Low, Close)
and are intended for use in assessing the fidelity of synthetic data relative to real financial data.
"""

import numpy as np
from scipy.stats import kurtosis


def heavy_tails(data: np.ndarray) -> np.ndarray:
    """
    Compute excess kurtosis per sample, then average over samples for each channel.
    Returns:
        np.ndarray: Array of averaged excess kurtosis per channel.
    """
    R, l, N = data.shape
    kurt_vals = []
    for ch in range(N):
        sample_kurt = [kurtosis(data[r, :, ch], fisher=True) for r in range(R)]
        kurt_vals.append(np.mean(sample_kurt))
    return np.array(kurt_vals, dtype=float)

def autocorr_raw(data: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Compute lag-k autocorrelation per sample, then average across samples for each channel.
    """
    R, l, N = data.shape
    ac_vals = []

    for ch in range(N):
        sample_ac = []
        for r in range(R):
            x = data[r, :, ch]
            x_mean = x.mean()
            numerator = np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean))
            denominator = np.sum((x - x_mean)**2) + 1e-12
            sample_ac.append(numerator / denominator)
        ac_vals.append(np.mean(sample_ac))

    return np.array(ac_vals, dtype=float)

def volatility_clustering(data: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Compute lag-k autocorrelation of squared returns per sample,
    then average across samples for each channel.
    """
    R, l, N = data.shape
    ac_sq_vals = []

    for ch in range(N):
        sample_ac = []
        for r in range(R):
            x = data[r, :, ch]
            x_sq = x ** 2
            x_mean = x_sq.mean()
            numerator = np.sum((x_sq[:-lag] - x_mean) * (x_sq[lag:] - x_mean))
            denominator = np.sum((x_sq - x_mean) ** 2) + 1e-12
            sample_ac.append(numerator / denominator)
        ac_sq_vals.append(np.mean(sample_ac))

    return np.array(ac_sq_vals, dtype=float)
