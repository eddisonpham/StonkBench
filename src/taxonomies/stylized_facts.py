"""
Stylized Facts of Financial Time Series (Cont, 2001)

This module implements quantitative measures for the principal "stylized facts" observed in financial asset returns, following:
Rama Cont, "Empirical properties of asset returns: stylized facts and statistical issues." Quantitative Finance, 2001.

Implemented stylized facts and their metrics:

- Absence of Linear Autocorrelations: Measured as the average autocorrelation of raw returns at nonzero lags.
- Volatility Clustering: Measured as the average autocorrelation of squared returns at short lags.
- Long Memory in Volatility: Estimated as the decay exponent (beta) of autocorrelation in absolute returns.

All metrics are computed as averages across sample time series.
"""

import numpy as np

def autocorr_returns(data, lag=1):
    """
    Average lag-k autocorrelation of raw returns across samples.
    This computes the linear autocorrelation on raw returns to assess the absence of linear dependence.
    data: np.ndarray of shape (n_samples, length)
    """
    acfs = []
    for sample in data:
        r = sample
        r_mean = r.mean()
        numerator = np.sum((r[:-lag] - r_mean) * (r[lag:] - r_mean))
        denominator = np.sum((r - r_mean) ** 2)
        acfs.append(numerator / denominator if denominator != 0 else np.nan)
    acfs = [a for a in acfs if not np.isnan(a)]
    return np.mean(acfs) if len(acfs) > 0 else np.nan

def volatility_clustering(data, max_lag=1):
    """
    Average lag-k autocorrelation of *squared* returns across samples and lags 1...max_lag.
    This captures volatility clustering (persistence in variance), as in the stylized facts literature.
    data: np.ndarray of shape (n_samples, length)
    Returns: array of mean autocorrelations for lags 1,...,max_lag (length = max_lag)
    """
    acf_by_lag = []
    for lag in range(1, max_lag + 1):
        lag_acfs = []
        for sample in data:
            squared_r = sample ** 2
            m = squared_r.mean()
            num = np.sum((squared_r[:-lag] - m) * (squared_r[lag:] - m))
            den = np.sum((squared_r - m) ** 2)
            lag_acfs.append(num / den if den != 0 else np.nan)
        lag_acfs = [a for a in lag_acfs if not np.isnan(a)]
        acf_by_lag.append(np.mean(lag_acfs) if len(lag_acfs) > 0 else np.nan)

    return np.array(acf_by_lag)

def long_memory_volatility(data, max_lag=52):
    """
    Estimate the long memory of volatility by fitting the power-law decay of the autocorrelation of absolute returns.
    The decay exponent (beta) is estimated for each sample, and the mean is returned.
    """
    _, n_len = data.shape
    betas = []
    for sample in data:
        abs_r = np.abs(sample)
        mean = abs_r.mean()
        var = np.sum((abs_r - mean)**2) / n_len
        lags = np.arange(1, min(max_lag, n_len // 4))
        acf_vals = []
        for lag in lags:
            cov = np.sum((abs_r[:-lag] - mean) * (abs_r[lag:] - mean)) / (n_len - lag)
            acf_vals.append(cov / var)
        acf_vals = np.array(acf_vals)
        mask = acf_vals > 0
        if np.sum(mask) < 2:
            continue
        lags = lags[mask]
        acf_vals = acf_vals[mask]
        beta = -np.polyfit(np.log(lags), np.log(acf_vals), 1)[0]
        betas.append(beta)
    return np.mean(betas)
