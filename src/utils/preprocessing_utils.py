"""
Preprocessing utility functions
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.stattools import acf, pacf
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
    PyTorch Dataset for 3D time series data (R, l, N):
    - R: number of sequences/windows
    - l: sequence length
    - N: number of features

    Features:
    - Optional shuffling (only recommended for training)
    - Seed support for reproducibility
    - Optional transform on each sequence
    """
    def __init__(self, data: np.ndarray, shuffle: bool = False, seed: int = 42, transform=None):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        if data.ndim != 3:
            raise ValueError(f"Data must be 3D with shape (R, l, N), got {data.shape}")

        self.data = torch.from_numpy(data).float()
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed

        self._initialize_indices()

    def _initialize_indices(self):
        """Initialize or reshuffle indices based on the shuffle flag."""
        self.indices = list(range(len(self.data)))
        if self.shuffle:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.shuffle(self.indices)

    def __len__(self):
        """Return the number of sequences/windows."""
        return len(self.data)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.data[actual_idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def set_seed(self, seed: int):
        """Reset seed and reshuffle indices if shuffle=True."""
        self.seed = seed
        if self.shuffle:
            self._initialize_indices()

    def get_original_indices(self):
        """Return a copy of the current indices (shuffled or sequential)."""
        return self.indices.copy()

def create_dataloaders(
    train_data, 
    valid_data, 
    test_data,
    batch_size=32, 
    train_seed=None, 
    valid_seed=None,
    test_seed=None,
    num_workers=0, 
    pin_memory=False
):
    """
    Create train/validation DataLoaders for time series data.
    """
    train_dataset = TimeSeriesDataset(train_data, seed=train_seed, shuffle=True)
    valid_dataset = TimeSeriesDataset(valid_data, seed=valid_seed, shuffle=False)
    test_dataset = TimeSeriesDataset(test_data, seed=test_seed, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def find_length(data):
    """
    Find the length of the time series segment using PACF.
    """
    data = np.asarray(data)
    data = data[:min(20000, len(data))]
    
    nobs, nchan = data.shape
    if nobs < 20:
        raise ValueError("Too few observations to compute PACF.")

    max_lag = 0
    nlags = int(min(10 * np.log10(nobs), nobs - 1))
    
    for ch in range(nchan):
        pacf_vals = pacf(data[:, ch], nlags=nlags, method='yw')
        peaks = argrelextrema(pacf_vals, np.greater)[0]
        if len(peaks) > 0:
            lag = peaks[np.argmax(pacf_vals[peaks])]
            max_lag = max(max_lag, lag)

    if max_lag == 0:
        raise ValueError("No significant PACF peaks found; series may be nearly white noise.")

    return int(max_lag)

def sliding_window_view(data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    """
    Segment a 2D time series (L, N) into overlapping windows (R, window_size, N).
    """
    assert data.ndim == 2, "Input array must be 2D"
    L, N = data.shape
    assert L >= window_size, "Window size must be <= sequence length"

    num_windows = (L - window_size) // step + 1
    new_strides = (data.strides[0] * step, data.strides[0], data.strides[1])
    new_shape = (num_windows, window_size, N)
    return np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)

def _preprocess_parametric(ori_data, valid_ratio=0.1, test_ratio=0.1):
    """
    Preprocessing for parametric models: split full series into train/val/test.
    """
    ori_data = torch.from_numpy(ori_data)
    L = ori_data.shape[0]
    train_end = int(L * (1 - valid_ratio - test_ratio))
    valid_end = int(L * (1 - test_ratio))
    train_data = ori_data[:train_end]
    valid_data = ori_data[train_end:valid_end]
    test_data = ori_data[valid_end:]
    return train_data, valid_data, test_data

def _preprocess_non_parametric(ori_data, seq_length, valid_ratio=0.1, test_ratio=0.1, step=1, seed=42):
    """
    Preprocessing for non-parametric models: sliding windows and train/val/test split.
    """
    data = sliding_window_view(ori_data, seq_length, step=step)
    L = data.shape[0]
    train_end = int(L * (1 - valid_ratio - test_ratio))
    valid_end = int(L * (1 - test_ratio))
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    return train_data, valid_data, test_data

def preprocess_data(cfg, supress_cfg_message=False):
    """
    Preprocess time series data for parametric or non-parametric models.
    Returns: train, valid, test
    """
    if not supress_cfg_message:
        show_with_start_divider(f"Preprocessing data for {cfg.get('ticker')}")
    
    ori_data_path = cfg.get('original_data_path')
    seq_length = cfg.get('seq_length', None)
    valid_ratio = cfg.get('valid_ratio', 0.1)
    test_ratio = cfg.get('test_ratio', 0.1)
    do_transformation = cfg.get('do_transformation', True)
    is_parametric = cfg.get('is_parametric', False)
    seed = cfg.get('seed', 42)

    if not os.path.exists(ori_data_path):
        show_with_end_divider(f"File {ori_data_path} does not exist.")
        return None

    df = pd.read_csv(ori_data_path)
    ori_data = df[REQUIRED_COLUMNS].values

    if do_transformation:
        scaler = LogReturnTransformation()
        ori_data = scaler.transform(ori_data)

    if seq_length is None:
        seq_length = find_length(ori_data)

    if is_parametric:
        return _preprocess_parametric(ori_data, valid_ratio, test_ratio)
    return _preprocess_non_parametric(ori_data, seq_length, valid_ratio, test_ratio, seed=seed)