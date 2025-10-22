"""
Preprocessing utility functions
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


class LogReturnTransformation:
    """
    Transform a price series into log returns and provide an inverse transformation
    to reconstruct the original prices.
    """
    def transform(self, data):
        """
        Compute log returns.
        """
        data = np.asarray(data)
        return np.log(data[1:] / data[:-1])

    def inverse_transform(self, log_returns, initial_value):
        """
        Reconstruct the original price series from log returns.

        Args:
            log_returns (np.ndarray): Log returns of shape (L-1, N)
            initial_value (np.ndarray or float): Initial price(s)

        Returns:
            np.ndarray: Reconstructed price series of shape (L, N)
        """
        prices = [np.asarray(initial_value)]
        for r in log_returns:
            prices.append(prices[-1] * np.exp(r))
        return np.vstack(prices)

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

def _preprocess_parametric(ori_data, valid_ratio):
    """
    Preprocessing steps for parametric models.
    """
    idx = np.arange(ori_data.shape[0])
    np.random.shuffle(idx)
    ori_data = ori_data[idx]
    ori_data = torch.from_numpy(ori_data)
    print('Data shape:', tuple(ori_data.size()))
    split = int(ori_data.shape[0] * (1 - valid_ratio))
    show_with_end_divider(f'Preprocessing for parametric models done.')
    return ori_data[:split], ori_data[split:]

def _preprocess_non_parametric(ori_data, seq_length, valid_ratio, seed=None):
    """
    Preprocessing steps for non-parametric models.
    """
    data = sliding_window_view(ori_data, seq_length)  # (R, l, N)
    print('Data shape:', data.shape)
    
    np.random.shuffle(data)
    split = int(data.shape[0] * (1 - valid_ratio))
    train_data, valid_data = data[:split], data[split:]
    show_with_end_divider(f'Preprocessing for non-parametric models done.')
    return train_data, valid_data

def preprocess_data(cfg, supress_cfg_message = False):
    """
    Preprocess time series data for parametric or non-parametric models.

    Args:
        cfg (dict): Configuration options:
            - original_data_path (str): Path to input CSV file.
            - seq_length (int, optional): Sub-sequence length; auto-determined if None.
            - valid_ratio (float, optional): Validation split ratio (default 0.1).
            - do_transformation (bool, optional): Apply log return transformation (default True).
            - is_parametric (bool, optional): If True, skip segmentation (default False).
            - seed (int, optional): Random seed for reproducibility.
        supress_cfg_message (bool): If True, suppress preprocessing logs.

    Returns:
        tuple: (train_data, valid_data) as np.ndarray or torch.Tensor, or None on failure.
    """
    if not supress_cfg_message:
        show_with_start_divider(f"Data preprocessing with settings:{cfg}")

    ori_data_path = cfg.get('original_data_path', None)
    seq_length = cfg.get('seq_length', None)
    valid_ratio = cfg.get('valid_ratio', 0.1)
    do_transformation = cfg.get('do_transformation', True)
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
        ori_data = df[REQUIRED_COLUMNS].values  # (L, N)
    except Exception as e:
        show_with_end_divider(f"Error: An error occurred during reading data: {e}.")
        return None

    if seq_length is None:
        seq_length = int(np.mean(np.apply_along_axis(find_length, 0, ori_data)))

    if do_transformation:
        scaler = LogReturnTransformation()
        ori_data = scaler.transform(ori_data)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if is_parametric:
        return _preprocess_parametric(ori_data, valid_ratio)
    return _preprocess_non_parametric(ori_data, seq_length, valid_ratio)
