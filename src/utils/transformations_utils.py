"""
Transformation utilities for data processing and DataLoader creation.

This module provides utilities for creating PyTorch DataLoaders and other
data transformation operations needed for model training.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional

def create_dataloaders(
    train_data: np.ndarray,
    valid_data: np.ndarray,
    batch_size: int = 32,
    train_seed: int = 42,
    valid_seed: int = 123,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
    shuffle_valid: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from numpy arrays.
    
    Args:
        train_data (np.ndarray): Training data of shape (R, l, N)
        valid_data (np.ndarray): Validation data of shape (R, l, N)
        batch_size (int): Batch size for DataLoaders
        train_seed (int): Random seed for training DataLoader
        valid_seed (int): Random seed for validation DataLoader
        num_workers (int): Number of worker processes for data loading
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        shuffle_train (bool): Whether to shuffle training data
        shuffle_valid (bool): Whether to shuffle validation data
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders
    """
    # Convert numpy arrays to torch tensors
    train_tensor = torch.FloatTensor(train_data)
    valid_tensor = torch.FloatTensor(valid_data)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(train_tensor)
    valid_dataset = TensorDataset(valid_tensor)
    
    # Create generators for reproducible shuffling
    train_generator = torch.Generator()
    train_generator.manual_seed(train_seed)
    
    valid_generator = torch.Generator()
    valid_generator.manual_seed(valid_seed)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=train_generator if shuffle_train else None
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=shuffle_valid,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=valid_generator if shuffle_valid else None
    )
    
    return train_loader, valid_loader

def normalize_data(data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, dict]:
    """
    Normalize data using specified method.
    
    Args:
        data (np.ndarray): Input data to normalize
        method (str): Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Tuple[np.ndarray, dict]: Normalized data and normalization parameters
    """
    if method == 'minmax':
        data_min = np.min(data, axis=(0, 1), keepdims=True)
        data_max = np.max(data, axis=(0, 1), keepdims=True)
        
        # Avoid division by zero
        range_val = data_max - data_min
        range_val[range_val == 0] = 1
        
        normalized_data = (data - data_min) / range_val
        params = {'min': data_min, 'max': data_max, 'method': method}
        
    elif method == 'zscore':
        data_mean = np.mean(data, axis=(0, 1), keepdims=True)
        data_std = np.std(data, axis=(0, 1), keepdims=True)
        
        # Avoid division by zero
        data_std[data_std == 0] = 1
        
        normalized_data = (data - data_mean) / data_std
        params = {'mean': data_mean, 'std': data_std, 'method': method}
        
    elif method == 'robust':
        data_median = np.median(data, axis=(0, 1), keepdims=True)
        data_mad = np.median(np.abs(data - data_median), axis=(0, 1), keepdims=True)
        
        # Avoid division by zero
        data_mad[data_mad == 0] = 1
        
        normalized_data = (data - data_median) / data_mad
        params = {'median': data_median, 'mad': data_mad, 'method': method}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_data, params

def denormalize_data(normalized_data: np.ndarray, params: dict) -> np.ndarray:
    """
    Denormalize data using stored parameters.
    
    Args:
        normalized_data (np.ndarray): Normalized data to denormalize
        params (dict): Normalization parameters from normalize_data()
        
    Returns:
        np.ndarray: Denormalized data
    """
    method = params['method']
    
    if method == 'minmax':
        range_val = params['max'] - params['min']
        denormalized_data = normalized_data * range_val + params['min']
        
    elif method == 'zscore':
        denormalized_data = normalized_data * params['std'] + params['mean']
        
    elif method == 'robust':
        denormalized_data = normalized_data * params['mad'] + params['median']
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return denormalized_data

def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
    drop_last: bool = False
) -> np.ndarray:
    """
    Create sliding windows from time series data.
    
    Args:
        data (np.ndarray): Input data of shape (T, N) where T is time, N is features
        window_size (int): Size of each window
        stride (int): Step size between windows
        drop_last (bool): Whether to drop the last incomplete window
        
    Returns:
        np.ndarray: Windowed data of shape (num_windows, window_size, N)
    """
    T, N = data.shape
    
    if window_size > T:
        raise ValueError(f"Window size {window_size} is larger than data length {T}")
    
    # Calculate number of windows
    num_windows = (T - window_size) // stride + 1
    
    if not drop_last and (T - window_size) % stride != 0:
        num_windows += 1
    
    # Create windows
    windows = []
    for i in range(0, T - window_size + 1, stride):
        window = data[i:i + window_size]
        windows.append(window)
    
    # Handle last incomplete window if not dropping it
    if not drop_last and (T - window_size) % stride != 0:
        last_start = T - window_size
        if last_start >= 0:
            last_window = data[last_start:T]
            if len(last_window) == window_size:
                windows.append(last_window)
    
    return np.array(windows)

def split_train_valid(
    data: np.ndarray,
    valid_ratio: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets.
    
    Args:
        data (np.ndarray): Input data to split
        valid_ratio (float): Ratio of data to use for validation
        seed (int, optional): Random seed for reproducible splits
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Training and validation data
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = len(data)
    valid_size = int(num_samples * valid_ratio)
    
    # Create random indices
    indices = np.random.permutation(num_samples)
    
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]
    
    train_data = data[train_indices]
    valid_data = data[valid_indices]
    
    return train_data, valid_data