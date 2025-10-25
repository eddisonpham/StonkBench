"""
Data Conversion Utilities for Time Series

This module provides standardized conversion functions for time series data across
the benchmark framework. These utilities handle:
- Converting between numpy arrays and torch tensors
- Ensuring consistent 3D shape (R, l, N) where R=samples, l=timesteps, N=features
- Handling edge cases and validation

This version no longer drops any channel (such as a timestamp) automatically.
"""

import numpy as np
import torch


def to_torch_features_abc(data):
    """
    Convert input to torch tensor with shape (R, l, N).
    
    This function standardizes input data for PyTorch-based metrics and models by:
    1. Converting to torch.Tensor if necessary
    2. Ensuring 3D shape (R, l, N)
    """
    if not torch.is_tensor(data):
        data = torch.tensor(data)
    if data.ndim == 2:
        data = data.unsqueeze(0)
    elif data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")
    return data

def to_numpy_abc(data):
    """
    Convert input to numpy array with shape (R, l, N).
    
    This function standardizes input data for numpy-based metrics by:
    1. Converting to numpy.ndarray if necessary
    2. Ensuring 3D shape (R, l, N)
    """
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None, ...]
    elif data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    return data

def to_numpy_features_for_visualization(data):
    """
    Convert data to numpy array format suitable for visualization.
    
    This function is specifically designed for visualization functions that need to:
    convert any input to numpy and handle 2D or 3D input flexibly.
    Shape will be (R, l, N).
    """
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    
    if data.ndim == 2:
        data = data[None, ...]
    
    return data

