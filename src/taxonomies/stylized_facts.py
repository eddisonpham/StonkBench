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

def leverage_effect(data, lag=1):
    """
    Average correlation between return and next squared return across samples.
    """
    n_samples, n_len = data.shape
    cors = []
    for sample in data:
        r_t = sample[:-lag]
        r_next2 = sample[lag:]**2
        r_t_mean = r_t.mean()
        r_next2_mean = r_next2.mean()
        numerator = np.sum((r_t - r_t_mean)*(r_next2 - r_next2_mean))
        denominator = np.sqrt(np.sum((r_t - r_t_mean)**2) * np.sum((r_next2 - r_next2_mean)**2))
        cors.append(numerator / denominator)
    return np.mean(cors)

def long_memory_volatility(data, max_lag=100):
    """
    Estimate decay exponent beta for autocorrelation of absolute returns.
    Fits rho(tau) ~ tau^-beta.
    Returns average beta across samples.
    """
    def power_law(x, c, beta):
        return c * x**(-beta)

    n_samples, n_len = data.shape
    betas = []
    for sample in data:
        abs_r = np.abs(sample)
        acf_vals = []
        lags = np.arange(1, min(max_lag, n_len//2))
        abs_r_mean = abs_r.mean()
        var_r = np.sum((abs_r - abs_r_mean)**2)
        for lag in lags:
            cov = np.sum((abs_r[:-lag] - abs_r_mean)*(abs_r[lag:] - abs_r_mean))
            acf_vals.append(cov / var_r)
        acf_vals = np.array(acf_vals)
        try:
            popt, _ = curve_fit(power_law, lags, acf_vals, p0=(acf_vals[0], 0.5))
            betas.append(popt[1])
        except:
            continue
    return np.mean(betas)


