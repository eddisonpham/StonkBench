"""
Preprocessing utility functions

This module provides essential utility functions supporting the TSGBench standardized
preprocessing pipeline.

Key Features:
- MinMaxScaler: Implements the normalization step from TSGBench pipeline
- Path management and directory creation
- Display utilities for progress tracking
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.stattools import acf
from torch.utils.data import Dataset, DataLoader
from scipy.signal import argrelextrema

from src.utils.display_utils import (
    show_divider,
    show_with_start_divider,
    show_with_end_divider
)

REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close']


class MinMaxScaler():
    """
    Min-Max normalization scaler implementing the TSGBench normalization step.
    
    This class implements the normalization step from the TSGBench preprocessing pipeline,
    which normalizes the dataset to the range [0, 1] to enhance efficiency and numerical
    stability. This is the final step in the TSGBench pipeline, applied after segmentation,
    shuffling, and train-test splitting.
    
    The normalization formula is: (x - min) / (max - min)
    The inverse transformation is: x * (max - min) + min
    """
    
    def fit_transform(self, data): 
        """
        Fit the scaler to the data and transform it in one step.
        
        Args:
            data (numpy.ndarray): Input data to fit and transform
            
        Returns:
            numpy.ndarray: Normalized data in range [0, 1]
        """
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):    
        """
        Compute the minimum and range values for normalization.
        
        Args:
            data (numpy.ndarray): Input data to compute statistics from
            
        Returns:
            self: Returns self for method chaining
        """
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self
        
    def transform(self, data):
        """
        Apply the normalization transformation to the data.
        
        Args:
            data (numpy.ndarray): Input data to normalize
            
        Returns:
            numpy.ndarray: Normalized data in range [0, 1]
        """
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data
    
    def inverse_transform(self, data):
        """
        Reverse the normalization transformation.
        
        Args:
            data (numpy.ndarray): Normalized data in range [0, 1]
            
        Returns:
            numpy.ndarray: Data transformed back to original scale
        """
        data *= self.range
        data += self.mini
        return data

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset class for time series data with seed support for reproducible batching.
    
    This class handles time series data with shape (R, l, N) where:
    - R: Number of sequences
    - l: Sequence length (time steps)  
    - N: Number of variables/features
    
    The dataset supports seed-based shuffling and proper batching for time series generation models.
    
    Args:
        data (numpy.ndarray): Time series data with shape (R, l, N)
        seed (int, optional): Random seed for reproducible shuffling (default: None)
        transform (callable, optional): Optional transform to apply to each sample
    """
    
    def __init__(self, data, seed=None, transform=None):
        """
        Initialize the TimeSeriesDataset.
        
        Args:
            data (numpy.ndarray): Time series data with shape (R, l, N)
            seed (int, optional): Random seed for reproducible shuffling
            transform (callable, optional): Optional transform to apply to each sample
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        
        if data.ndim != 3:
            raise ValueError(f"Data must be 3D with shape (R, l, N), got shape {data.shape}")

        self.data = torch.from_numpy(data).float()
        self.transform = transform

        self.indices = list(range(len(data)))
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.shuffle(self.indices)
        else:
            random.shuffle(self.indices)
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single time series sequence.
        
        Args:
            idx (int): Index of the sequence to retrieve
            
        Returns:
            torch.Tensor: Time series sequence with shape (l, N)
        """
        # Use shuffled indices for reproducible access
        actual_idx = self.indices[idx]
        sample = self.data[actual_idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_original_indices(self):
        """
        Get the original indices in shuffled order.
        
        Returns:
            list: List of original indices in the order they are accessed
        """
        return self.indices.copy()
    
    def set_seed(self, seed):
        """
        Reset the dataset with a new seed for different shuffling.
        
        Args:
            seed (int): New random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.indices = list(range(len(self.data)))
        random.shuffle(self.indices)

def create_dataloaders(
    train_data, 
    valid_data, 
    batch_size=32, 
    train_seed=None, 
    valid_seed=None,
    num_workers=0, 
    pin_memory=False
):
    """
    Create train/validation DataLoaders for time series data.

    Args:
        train_data, valid_data (np.ndarray): Arrays of shape (R, l, N)
        batch_size (int): Batch size (default=32)
        train_seed, valid_seed (int): Shuffle seeds
        num_workers (int): DataLoader workers
        pin_memory (bool): Pin memory for GPU (default=False)

    Returns:
        tuple: (train_loader, valid_loader)
    """
    train_dataset = TimeSeriesDataset(train_data, seed=train_seed)
    valid_dataset = TimeSeriesDataset(valid_data, seed=valid_seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader


def find_length(data, default_l=125):
    """
    Simple heuristic to find time series segment length using ACF peaks.
    """
    data = np.asarray(data).flatten()  # handle 1D or 2D
    data = data[:min(20000, len(data))]  # keep manageable size

    nobs = len(data)
    if nobs < 20:
        return default_l

    nlags = int(min(10 * np.log10(nobs), nobs - 1))
    auto_corr = acf(data, nlags=nlags, fft=True)[3:]
    local_max = argrelextrema(auto_corr, np.greater)[0]

    if len(local_max) == 0:
        return default_l

    best = local_max[np.argmax(auto_corr[local_max])]
    if 3 <= best <= 300:
        return int(best + 3)
    return default_l

def sliding_window_view(data, window_size, step=1):
    """
    Segment a 2D time series (L, N) into overlapping windows (R, l, N).

    Args:
        data (np.ndarray): Input array of shape (L, N)
        window_size (int): Length of each segment (l)
        step (int): Stride between windows (default=1)

    Returns:
        np.ndarray: 3D array of shape (R, l, N)
    """
    assert data.ndim == 2, "Input array must be 2D"
    L, N = data.shape
    assert L >= window_size, "Window size must be <= sequence length"

    B = L - window_size + 1
    new_shape = (B, window_size, N)
    new_strides = (data.strides[0],) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)

def preprocess_data(cfg, supress_cfg_message = False):
    """
    Preprocess time series data for parametric or non-parametric models.

    Args:
        cfg (dict): Configuration options:
            - original_data_path (str): Path to input CSV file.
            - seq_length (int, optional): Sub-sequence length; auto-determined if None.
            - valid_ratio (float, optional): Validation split ratio (default 0.1).
            - do_normalization (bool, optional): Apply normalization (default True).
            - is_parametric (bool, optional): If True, skip segmentation (default False).
            - seed (int, optional): Random seed for reproducibility.
        supress_cfg_message (bool): If True, suppress preprocessing logs.

    Returns:
        tuple: (train_data, valid_data) as np.ndarray or torch.Tensor, or None on failure.
    """
    if not supress_cfg_message:
        show_with_start_divider(f"Data preprocessing with settings:{cfg}")

    ori_data_path = cfg.get('original_data_path',None)
    seq_length = cfg.get('seq_length',None)
    valid_ratio = cfg.get('valid_ratio',0.1)
    do_normalization = cfg.get('do_normalization',True)
    is_parametric = cfg.get('is_parametric', False)
    seed = cfg.get('seed', None)

    if not os.path.exists(ori_data_path):
        curr_dir = os.getcwd()
        print(f"Current working directory: {curr_dir}")
        show_with_end_divider(f'Original file path {ori_data_path} does not exist.')
        return None
    
    _, ext = os.path.splitext(ori_data_path)
    try:
        if ext != '.csv':
            show_with_end_divider(f"Error: Unsupported file extension: {ext}")
            return None
        df = pd.read_csv(ori_data_path)
        ori_data = df[REQUIRED_COLUMNS].values # (L, N)
    except Exception as e:
        show_with_end_divider(f"Error: An error occurred during reading data: {e}.")
        return None

    if is_parametric:
        ori_data = torch.from_numpy(ori_data)
        print('Data shape:', tuple(ori_data.size()))
        split = int(ori_data.shape[0] * (1 - valid_ratio))

        show_with_end_divider(f'Preprocessing for parametric models done.')
        return ori_data[:split], ori_data[split:]

    if seq_length is None:
        seq_length = int(np.mean(np.apply_along_axis(find_length, 0, ori_data)))
    
    data = sliding_window_view(ori_data, seq_length) # (R, l, N)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    np.random.shuffle(data) # (R', l, N)

    print('Data shape:', data.shape)
    split = int(data.shape[0] * (1 - valid_ratio))
    train_data, valid_data = data[:split], data[split:]

    if do_normalization:
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        valid_data = scaler.transform(valid_data)

    show_with_end_divider(f'Preprocessing for non-parametric models done.')
    return train_data, valid_data
