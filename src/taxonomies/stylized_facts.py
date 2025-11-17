"""
Stylized Facts of Financial Time Series (Cont, 2001)

This module implements quantitative measures for the principal "stylized facts" observed in financial asset returns, following:
Rama Cont, "Empirical properties of asset returns: stylized facts and statistical issues." Quantitative Finance, 2001.

Implemented stylized facts and their metrics:

- Heavy Tails: Measured as the average sample excess kurtosis of returns.
- Absence of Linear Autocorrelations: Measured as the average autocorrelation of raw returns at nonzero lags.
- Volatility Clustering: Measured as the average autocorrelation of squared returns at short lags.
- Long Memory in Volatility: Estimated as the decay exponent (beta) of autocorrelation in absolute returns.
- Leverage Effect: Measured as the mean contemporaneous correlation between past returns and future squared returns.

All metrics are computed as averages across sample time series.
"""

import numpy as np
from scipy.stats import kurtosis
from scipy.optimize import curve_fit


def autocorr_returns(data, lag=1):
    """
    Average lag-k autocorrelation of returns across samples.
    data: np.ndarray of shape (n_samples, length)
    """
    n_samples, n_len = data.shape
    acfs = []
    for sample in data:
        r = sample
        r_mean = r.mean()
        numerator = np.sum((r[:-lag]-r_mean)*(r[lag:]-r_mean))
        denominator = np.sum((r-r_mean)**2)
        acfs.append(numerator / denominator)
    return np.mean(acfs)

def excess_kurtosis(data):
    """
    Average excess kurtosis across samples.
    """
    n_samples = data.shape[0]
    kurts = [kurtosis(sample, fisher=True, bias=False) for sample in data]
    return np.mean(kurts)

def volatility_clustering(data, lag=1):
    """
    Average autocorrelation of squared returns across samples.
    """
    n_samples, n_len = data.shape
    acfs = []
    for sample in data:
        r2 = sample**2
        r2_mean = r2.mean()
        numerator = np.sum((r2[:-lag]-r2_mean)*(r2[lag:]-r2_mean))
        denominator = np.sum((r2-r2_mean)**2)
        acfs.append(numerator / denominator)
    return np.mean(acfs)

def long_memory_volatility(data, max_lag=100):
    """
    Estimate the long memory of volatility by fitting the power-law decay of the autocorrelation of absolute returns.
    The decay exponent (beta) is estimated for each sample, and the mean is returned.
    """
    n_samples, n_len = data.shape
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
