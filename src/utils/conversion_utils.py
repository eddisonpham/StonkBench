"""
Data Conversion Utilities for Time Series

This module provides standardized conversion functions for time series data across
the benchmark framework. These utilities handle:
- Converting between numpy arrays and torch tensors
- Ensuring consistent 3D shape (A, B, C) where A=samples, B=timesteps, C=features
- Dropping timestamp channels when appropriate
- Handling edge cases and validation

The project convention assumes that when multiple channels exist, channel 0 is a
timestamp and should be dropped for metric calculations.
"""

import numpy as np
import torch


def to_torch_features_abc(data):
    """
    Convert input to torch tensor with shape (A, B, C_features) and drop leading timestamp channel if present.
    
    This function standardizes input data for PyTorch-based metrics and models by:
    1. Converting to torch.Tensor if necessary
    2. Ensuring 3D shape (A, B, C)
    3. Dropping the timestamp channel at index 0 if C >= 2
    
    Args:
        data: Input data (numpy.ndarray or torch.Tensor) of shape (B, C) or (A, B, C)
        
    Returns:
        torch.Tensor: Data of shape (A, B, C_features) with timestamp channel removed
        
    Raises:
        ValueError: If input is not 2D or 3D, or if only timestamp channel exists
    """
    # Convert to torch if needed
    if not torch.is_tensor(data):
        data = torch.tensor(data)
    
    # Ensure 3D shape
    if data.ndim == 2:
        # (B, C) -> (1, B, C)
        data = data.unsqueeze(0)
    elif data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")
    
    # Drop timestamp channel (assumed at index 0) if present
    # Heuristic: if C >= 2, always drop channel 0 as timestamp per project convention
    if data.shape[2] >= 2:
        data = data[:, :, 1:]
    else:
        # Nothing to compare if there is only 1 channel (timestamp only)
        raise ValueError("Input appears to have only a timestamp channel and no features.")
    
    return data

def to_numpy_abc(data):
    """
    Convert input to numpy array with shape (A, B, C) and drop leading timestamp channel if present.
    
    This function standardizes input data for numpy-based metrics by:
    1. Converting to numpy.ndarray if necessary
    2. Ensuring 3D shape (A, B, C)
    3. Dropping the timestamp channel at index 0 if C >= 2
    
    Args:
        data: Input data (numpy.ndarray or torch.Tensor) of shape (B, C) or (A, B, C)
        
    Returns:
        np.ndarray: Data of shape (A, B, C) with timestamp channel removed
        
    Raises:
        ValueError: If input is not 2D or 3D, or if only timestamp channel exists
    """
    # Convert to numpy
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)

    # Ensure 3D shape
    if data.ndim == 2:
        # (B, C) -> (1, B, C)
        data = data[None, ...]
    elif data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    # Drop timestamp channel (assumed at index 0) if present
    # Heuristic: if C >= 2, always drop channel 0 as timestamp per project convention
    if data.shape[2] >= 2:
        data = data[:, :, 1:]
    else:
        # Nothing to compare if there is only 1 channel (timestamp only)
        raise ValueError("Input appears to have only a timestamp channel and no features.")
    
    return data

def to_numpy_features_for_visualization(data):
    """
    Convert data to numpy array format suitable for visualization, dropping timestamp channel.
    
    This function is specifically designed for visualization functions that need to:
    1. Convert any input to numpy
    2. Handle 2D or 3D input flexibly
    3. Drop timestamp channels for cleaner visualizations
    
    Args:
        data: Input data (numpy.ndarray or torch.Tensor) of shape (B, C) or (A, B, C)
        
    Returns:
        np.ndarray: Data of shape (A, B, C_features) with timestamp channel removed if present
    """
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    
    # Ensure 3D shape
    if data.ndim == 2:
        data = data[None, ...]
    
    # Drop timestamp channel if present (more lenient for visualization)
    if data.shape[2] >= 2:
        data = data[:, :, 1:]
    
    return data

