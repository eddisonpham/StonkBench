"""
Stylized Facts Metrics for Financial Time Series

This module implements quantitative metrics for evaluating the presence of stylized facts
in financial time series data. Stylized facts are statistical properties commonly observed
in real-world financial markets, such as:

- Heavy tails (excess kurtosis) in returns
- Absence of autocorrelation in raw returns
- Volatility clustering (autocorrelation in squared returns)
- Long memory in absolute returns
- Non-stationarity in price and volume series

The provided functions operate on multivariate time series data (e.g., Open, High, Low, Close, Adj Close, Volume)
and are intended for use in assessing the fidelity of synthetic data relative to real financial data.
"""

import numpy as np
from scipy.stats import kurtosis

PRICE_IDX = [0, 1, 2, 3, 4]  # Open, High, Low, Close, Adj Close
RELEVANT_PRICE_IDX = [3, 4]  # Close, Adj Close
VOLUME_IDX = [5]             # Volume


def log_returns(data: np.ndarray, channels: list = PRICE_IDX) -> np.ndarray:
    """
    Compute log returns for selected price channels.

    Args:
        data (np.ndarray): Array of shape (R, l, N)
        channels (list): Indices of price channels to compute log returns

    Returns:
        np.ndarray: Same shape array with log returns for selected channels
    """
    data_ret = np.copy(data)
    for ch in channels:
        data_ret[:, 1:, ch] = np.log(data[:, 1:, ch]) - np.log(data[:, :-1, ch])
        data_ret[:, 0, ch] = 0.0
    return data_ret


def heavy_tails(data: np.ndarray) -> np.ndarray:
    """
    Excess kurtosis (heavy tails) for Close and Adj Close.

    Args:
        data (np.ndarray): Array of shape (R, l, 6)

    Returns:
        np.ndarray: Excess kurtosis per channel (Close, Adj Close)
    """
    data_ret = log_returns(data, PRICE_IDX)
    kurt_vals = []
    for ch in RELEVANT_PRICE_IDX:
        x = data_ret[:, :, ch].flatten()
        kurt_vals.append(kurtosis(x, fisher=True))
    return np.array(kurt_vals)


def autocorr_raw(data: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Lag-1 autocorrelation of raw log returns for Close and Adj Close.

    Args:
        data (np.ndarray): Array of shape (R, l, 6)
        lag (int): Lag for autocorrelation

    Returns:
        np.ndarray: Autocorrelation per channel
    """
    data_ret = log_returns(data, PRICE_IDX)
    ac_vals = []
    for ch in RELEVANT_PRICE_IDX:
        x = data_ret[:, :, ch].flatten()
        x_mean = np.mean(x)
        numerator = np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean))
        denominator = np.sum((x - x_mean) ** 2)
        ac_vals.append(numerator / denominator)
    return np.array(ac_vals)


def volatility_clustering(data: np.ndarray) -> np.ndarray:
    """
    Lag-1 autocorrelation of squared log returns for Close and Adj Close.

    Args:
        data (np.ndarray): Array of shape (R, l, 6)

    Returns:
        np.ndarray: Autocorrelation of squared returns per channel
    """
    data_ret = log_returns(data, PRICE_IDX)
    ac_sq_vals = []
    for ch in RELEVANT_PRICE_IDX:
        x = data_ret[:, :, ch].flatten()
        x_sq = x ** 2
        x_mean = np.mean(x_sq)
        numerator = np.sum((x_sq[:-1] - x_mean) * (x_sq[1:] - x_mean))
        denominator = np.sum((x_sq - x_mean) ** 2)
        ac_sq_vals.append(numerator / denominator)
    return np.array(ac_sq_vals)


def long_memory_abs(data: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """
    Average autocorrelation of absolute log returns for Close and Adj Close.

    Args:
        data (np.ndarray): Array of shape (R, l, 6)
        max_lag (int): Maximum lag to compute

    Returns:
        np.ndarray: Average autocorrelation per channel
    """
    data_ret = log_returns(data, PRICE_IDX)
    avg_ac_abs = []
    for ch in RELEVANT_PRICE_IDX:
        x = np.abs(data_ret[:, :, ch].flatten())
        ac_vals = []
        for lag in range(1, min(max_lag + 1, len(x))):
            x_mean = np.mean(x)
            numerator = np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean))
            denominator = np.sum((x - x_mean) ** 2)
            ac_vals.append(numerator / denominator)
        avg_ac_abs.append(np.mean(ac_vals))
    return np.array(avg_ac_abs)


def non_stationarity(data: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Non-stationarity via coefficient of variation of rolling variance.
    Applied to Close, Adj Close, and Volume.

    Args:
        data (np.ndarray): Array of shape (R, l, 6)
        window (int): Rolling window length

    Returns:
        np.ndarray: Non-stationarity measure per channel
    """
    channels = RELEVANT_PRICE_IDX + VOLUME_IDX  # Close, Adj Close, Volume
    nonstat_vals = []
    for ch in channels:
        x = data[:, :, ch].flatten()
        if len(x) < window:
            nonstat_vals.append(np.nan)
            continue
        rolling_var = [np.var(x[i:i + window]) for i in range(len(x) - window + 1)]
        nonstat_vals.append(np.std(rolling_var) / (np.mean(rolling_var) + 1e-8))
    return np.array(nonstat_vals)
