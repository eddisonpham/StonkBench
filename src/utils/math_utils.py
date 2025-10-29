"""
Mathematical and statistical utility functions.

This module provides utilities for mathematical computations and statistical operations.
"""

import torch
import numpy as np
from typing import Tuple


def histogram_torch(x, n_bins, density=True):
    """
    Compute histogram of a tensor using PyTorch.

    Args:
        x (torch.Tensor): Input tensor (N, 1).
        n_bins (int): Number of histogram bins.
        density (bool): Whether to normalize to a density.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (histogram counts, bin edges)
    """
    a, b = x.min().item(), x.max().item()
    b = b + 1e-5 if b == a else b
    bins = torch.linspace(a, b, n_bins + 1)
    delta = bins[1] - bins[0]
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """
    Compute autocorrelation function (ACF) up to max_lag.

    Args:
        x (torch.Tensor): Input data of shape (batch, time, features).
        max_lag (int): Maximum lag to compute.
        dim (tuple): Dimensions to average over.

    Returns:
        torch.Tensor: ACF values.
    """
    acf_list = []
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def non_stationary_acf_torch(X, symmetric=False):
    """
    Compute non-stationary autocorrelation function.
    
    Args:
        X (torch.Tensor): Input data of shape (batch, time, features).
        symmetric (bool): Whether to compute symmetric correlations.
        
    Returns:
        torch.Tensor: Correlation matrix.
    """
    # Get the batch size, sequence length, and input dimension from the input tensor
    B, T, D = X.shape

    # Create a tensor to hold the correlations
    correlations = torch.zeros(T, T, D)

    # Loop through each time step from lag to T-1
    for t in range(T):
        # Loop through each lag from 1 to lag
        for tau in range(t, T):
            # Compute the correlation between X_{t, d} and X_{t-tau, d}
            correlation = torch.sum(X[:, t, :] * X[:, tau, :], dim=0) / (
                torch.norm(X[:, t, :], dim=0) * torch.norm(X[:, tau, :], dim=0))
            # Store the correlation in the output tensor
            correlations[t, tau, :] = correlation
            if symmetric:
                correlations[tau, t, :] = correlation

    return correlations


def skew_torch(x, dim=(0, 1), dropdims=True):
    """
    Compute skewness of a tensor.
    
    Args:
        x (torch.Tensor): Input tensor.
        dim (tuple): Dimensions to compute over.
        dropdims (bool): Whether to drop dimensions.
        
    Returns:
        torch.Tensor: Skewness values.
    """
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        # Remove the dimensions that were averaged over
        skew = skew.squeeze()
    return skew


def kurtosis_torch(x, dim=(0, 1), excess=True, dropdims=True):
    """
    Compute kurtosis of a tensor.
    
    Args:
        x (torch.Tensor): Input tensor.
        dim (tuple): Dimensions to compute over.
        excess (bool): Whether to compute excess kurtosis.
        dropdims (bool): Whether to drop dimensions.
        
    Returns:
        torch.Tensor: Kurtosis values.
    """
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        # Remove the dimensions that were averaged over
        kurtosis = kurtosis.squeeze()
    return kurtosis


def acf_diff(x):
    """
    Compute ACF difference metric.
    """
    return torch.sqrt(torch.pow(x, 2).sum(0))
