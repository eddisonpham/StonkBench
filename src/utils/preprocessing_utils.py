"""
Preprocessing utility functions
"""

from typing import Tuple
import os
import random
import pandas as pd
import torch
from statsmodels.tsa.stattools import pacf
from torch.utils.data import Dataset, DataLoader

from src.utils.display_utils import (
    show_with_start_divider,
    show_with_end_divider
)


class LogReturnTransformation:

    def transform(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log returns and preserve the initial value.
        Assumes data is a torch tensor.
        """
        log_returns = torch.log(data[1:] / data[:-1])
        initial_value = data[0]
        return log_returns, initial_value

    def inverse_transform(self, log_returns: torch.Tensor, initial_value: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the original price series from log returns.
        Assumes log_returns and initial_value are torch tensors or convertibles.
        """
        prices = [initial_value]
        for r in log_returns:
            prices.append(prices[-1] * torch.exp(r))
        return torch.stack(prices)

class TimeSeriesDataset(Dataset):

    def __init__(
        self, 
        data: torch.Tensor, 
        initial_values: torch.Tensor,
        shuffle: bool = False, 
        seed: int = 42, 
        transform=None, 
    ):
        self.data = data.float()
        self.initial_values = initial_values
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        actual_idx = self.indices[idx]
        sample = self.data[actual_idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.initial_values[actual_idx]

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
    pin_memory=False,
    train_initial=None,
    valid_initial=None,
    test_initial=None
):
    """
    Create train/validation DataLoaders for time series data.
    """
    train_dataset = TimeSeriesDataset(train_data, initial_values=train_initial, seed=train_seed, shuffle=True)
    valid_dataset = TimeSeriesDataset(valid_data, initial_values=valid_initial, seed=valid_seed, shuffle=False)
    test_dataset = TimeSeriesDataset(test_data, initial_values=test_initial, seed=test_seed, shuffle=False)

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
    pacf_vals = torch.from_numpy(pacf_vals)
    desired_length = int(torch.argmax(pacf_vals[1:]) + 1)
    print(f"Desired time series sample length (lag with max PACF >0): {desired_length}")
    print(f"PACF at that lag: {pacf_vals[desired_length]}")
    return desired_length

def sliding_window_view(data: torch.Tensor, window_size: int, stride: int = 1) -> torch.Tensor:
    """
    Segment a 1D tensor into overlapping windows of size `window_size` with a given `stride`.
    """
    assert data.ndim == 1, "Input tensor must be 1D"
    L = data.shape[0]
    assert L >= window_size, "Window size must be <= sequence length"

    num_windows = (L - window_size) // stride + 1
    return data.as_strided(size=(num_windows, window_size), stride=(data.stride(0) * stride, data.stride(0)))

def preprocess_parametric(
    ori_data: torch.Tensor, 
    initial_value: torch.Tensor, 
    valid_ratio: float = 0.1, 
    test_ratio: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocessing for parametric models: split full series into train/val/test.
    """
    L = ori_data.shape[0]
    train_end = int(L * (1 - valid_ratio - test_ratio))
    valid_end = int(L * (1 - test_ratio))
    train_data = ori_data[:train_end]
    valid_data = ori_data[train_end:valid_end]
    test_data = ori_data[valid_end:]
    
    scaler = LogReturnTransformation()
    
    train_prices = scaler.inverse_transform(train_data, initial_value)
    train_initial = initial_value
    valid_initial = train_prices[-1]
    
    valid_prices = scaler.inverse_transform(valid_data, valid_initial)
    test_initial = valid_prices[-1]
    
    return train_data, valid_data, test_data, train_initial, valid_initial, test_initial

def preprocess_non_parametric(
    ori_data: torch.Tensor, 
    original_prices: torch.Tensor,
    seq_length: int = None, 
    valid_ratio: float = 0.1, 
    test_ratio: float = 0.1, 
    stride: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocessing for non-parametric models: transformation, window length selection, sliding windows, and train/val/test split.
    """
    if seq_length is None:
        seq_length = find_length(ori_data)

    windows = sliding_window_view(ori_data, seq_length, stride=stride)
    L = windows.shape[0]
    train_end = int(L * (1 - valid_ratio - test_ratio))
    valid_end = int(L * (1 - test_ratio))
    train_data = windows[:train_end]
    valid_data = windows[train_end:valid_end]
    test_data = windows[valid_end:]
    
    train_indices = torch.arange(0, train_end) * stride
    valid_indices = torch.arange(train_end, valid_end) * stride
    test_indices = torch.arange(valid_end, L) * stride
    
    train_initial = original_prices[train_indices]
    valid_initial = original_prices[valid_indices]
    test_initial = original_prices[test_indices]
    
    return train_data, valid_data, test_data, train_initial, valid_initial, test_initial

def preprocess_data(cfg, supress_cfg_message=False):
    """
    Preprocess time series data for parametric or non-parametric models.
    Returns: train, valid, test, train_initial, valid_initial, test_initial
    where *_initial are initial values for reconstructing prices from log returns.
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
    original_prices = df['Close'].values
    original_prices = torch.from_numpy(original_prices)

    scaler = LogReturnTransformation()
    log_returns, initial_value = scaler.transform(original_prices)

    if is_parametric:
        return preprocess_parametric(log_returns, initial_value, valid_ratio, test_ratio)
    return preprocess_non_parametric(
        log_returns, 
        original_prices,
        seq_length=seq_length,
        valid_ratio=valid_ratio, 
        test_ratio=test_ratio,
        stride=1,
    )