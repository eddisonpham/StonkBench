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


class LogReturnTransformation:
    """
    Transform a price series into log returns and provide an inverse transformation
    to reconstruct the original prices.
    When transforming, preserves the initial price(s) as part of the transformation output,
    so inverse transformation can accurately recover the original series.
    """
    
    def transform(self, data):
        """
        Compute log returns and preserve the initial value.

        Returns:
            Tuple[np.ndarray, Any]: 
                - log_returns of shape (L-1,)
                - initial_value (the first price)
        """
        data = np.asarray(data)
        log_returns = np.log(data[1:] / data[:-1])
        initial_value = data[0]
        return log_returns, initial_value

    def inverse_transform(self, log_returns, initial_value):
        """
        Reconstruct the original price series from log returns.

        Args:
            log_returns (np.ndarray): Log returns of shape (L-1)
            initial_value (np.ndarray or float): Initial price(s)

        Returns:
            np.ndarray: Reconstructed price series of shape (L,)
        """
        prices = [np.asarray(initial_value)]
        for r in log_returns:
            prices.append(prices[-1] * np.exp(r))
        return np.array(prices)

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for 2D time series data (R, l):
    - R: number of sequences/windows
    - l: sequence length

    Features:
    - Optional shuffling (only recommended for training)
    - Seed support for reproducibility (for shuffling)
    - Optional transform on each sequence
    """

    def __init__(self, data: np.ndarray, shuffle: bool = False, seed: int = 42, transform=None):
        if isinstance(data, torch.Tensor):
            if data.ndim != 2:
                raise ValueError(f"Data must be 2D with shape (R, l), got {data.shape}")
            self.data = data.float()
        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(f"Data must be 2D with shape (R, l), got {data.shape}")
            self.data = torch.from_numpy(data).float()
        else:
            raise ValueError("Data must be a numpy array or a torch tensor")

        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed

        self._initialize_indices()

    def _initialize_indices(self):
        """Initialize or reshuffle indices based on the shuffle flag."""
        self.indices = list(range(len(self.data)))
        if self.shuffle:
            rnd = random.Random(self.seed)
            rnd.shuffle(self.indices)

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
    Find the time series sample length using PACF.
    Picks the lag > 0 with the maximum PACF value.
    """
    series = data[:min(20000, len(data))]
    nobs = len(series)
    nlags = min(200, nobs // 10)
    pacf_vals = pacf(series, nlags=nlags, method='yw')
    desired_length = int(np.argmax(pacf_vals[1:]) + 1)
    print(f"Desired time series sample length (lag with max PACF >0): {desired_length}")
    print(f"PACF at that lag: {pacf_vals[desired_length]}")
    return desired_length

def sliding_window_view(data: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    Segment a 2D time series (L) into overlapping windows (R, window_size).
    """
    assert data.ndim == 1, "Input array must be 1D"
    L = data.shape[0]
    assert L >= window_size, "Window size must be <= sequence length"

    num_windows = (L - window_size) // stride + 1
    new_strides = (data.strides[0] * stride, data.strides[0])
    new_shape = (num_windows, window_size)
    return np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)

def preprocess_parametric(ori_data, valid_ratio=0.1, test_ratio=0.1):
    """
    Preprocessing for parametric models: split full series into train/val/test.
    No transformation is applied here.
    """
    ori_data = torch.from_numpy(ori_data)
    L = ori_data.shape[0]
    train_end = int(L * (1 - valid_ratio - test_ratio))
    valid_end = int(L * (1 - test_ratio))
    train_data = ori_data[:train_end]
    valid_data = ori_data[train_end:valid_end]
    test_data = ori_data[valid_end:]
    return train_data, valid_data, test_data

def preprocess_non_parametric(
    ori_data, 
    seq_length=None, 
    valid_ratio=0.1, 
    test_ratio=0.1, 
    stride=1
):
    """
    Preprocessing for non-parametric models: transformation, window length selection, sliding windows, and train/val/test split.
    """
    data = np.asarray(ori_data)

    if seq_length is None:
        seq_length = find_length(data)

    windows = sliding_window_view(data, seq_length, stride=stride)
    L = windows.shape[0]
    train_end = int(L * (1 - valid_ratio - test_ratio))
    valid_end = int(L * (1 - test_ratio))
    train_data = windows[:train_end]
    valid_data = windows[train_end:valid_end]
    test_data = windows[valid_end:]
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
    is_parametric = cfg.get('is_parametric', False)

    if not os.path.exists(ori_data_path):
        show_with_end_divider(f"File {ori_data_path} does not exist.")
        raise FileNotFoundError(f"File {ori_data_path} does not exist.")

    df = pd.read_csv(ori_data_path)
    ori_data = df['Close'].values

    scaler = LogReturnTransformation()
    ori_data, _ = scaler.transform(ori_data)

    if is_parametric:
        return preprocess_parametric(ori_data, valid_ratio, test_ratio)
    return preprocess_non_parametric(
            ori_data, 
            seq_length=seq_length,
            valid_ratio=valid_ratio, 
            test_ratio=test_ratio,
            stride=1,
        )