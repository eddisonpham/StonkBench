"""
Preprocessing utility functions
"""

from typing import Tuple
import os
import random
import pandas as pd
import torch
from statsmodels.tsa.stattools import acf
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


def create_dataloader(
    data: torch.Tensor,
    initial_values: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = None,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    Create a DataLoader for a single dataset split.
    
    Args:
        data: Time series data (2D tensor: num_windows, window_length) or (1D tensor for parametric)
        initial_values: Initial values for each window/sample
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader for the given split
    """
    
    dataset = TimeSeriesDataset(
        data, 
        initial_values=initial_values, 
        seed=seed, 
        shuffle=shuffle
    )
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )


def find_length(data):
    """
    Find the time series sample length using ACF.
    Picks the lag > 0 with the maximum ACF value.
    """
    nobs = len(data)
    nlags = min(200, nobs // 10)
    acf_vals = torch.from_numpy(acf(data, nlags=nlags)[1:])
    desired_length = int(torch.argmax(acf_vals) + 1)
    print(f"Desired time series sample length (lag with max ACF >0): {desired_length}")
    print(f"ACF at that lag: {acf_vals[desired_length - 1]}")
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
    log_returns: torch.Tensor, 
    original_prices: torch.Tensor,
    valid_ratio: float = 0.1, 
    test_ratio: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocessing for parametric models into train, valid, test splits"""

    L = log_returns.shape[0]
    train_end = int(L * (1 - valid_ratio - test_ratio))
    valid_end = int(L * (1 - test_ratio))
    
    train_data = log_returns[:train_end]
    valid_data = log_returns[train_end:valid_end]
    test_data = log_returns[valid_end:]
    
    scaler = LogReturnTransformation()
    train_prices = scaler.inverse_transform(train_data, original_prices[0])
    train_initial = original_prices[0]
    valid_initial = train_prices[-1]
    
    valid_prices = scaler.inverse_transform(valid_data, valid_initial)
    test_initial = valid_prices[-1]
    
    return train_data, valid_data, test_data, train_initial, valid_initial, test_initial


def preprocess_non_parametric(
    log_returns: torch.Tensor, 
    original_prices: torch.Tensor,
    valid_ratio: float = 0.1, 
    test_ratio: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocessing for non-parametric models into train, valid, test splits"""

    L = log_returns.shape[0]
    train_end = int(L * (1 - valid_ratio - test_ratio))
    valid_end = int(L * (1 - test_ratio))
    
    train_data = log_returns[:train_end]
    valid_data = log_returns[train_end:valid_end]
    test_data = log_returns[valid_end:]
    
    train_prices = original_prices[:train_end+1]
    valid_prices = original_prices[train_end:valid_end+1]
    test_prices = original_prices[valid_end:]
    
    return train_data, valid_data, test_data, train_prices[0], valid_prices[0], test_prices[0]


def preprocess_data(cfg, supress_cfg_message=False):
    """Preprocess time series data into train, valid, test splits"""

    if not supress_cfg_message:
        show_with_start_divider(f"Preprocessing data for {cfg.get('index')}")
    
    ori_data_path = cfg.get('original_data_path')
    valid_ratio = cfg.get('valid_ratio', 0.1)
    test_ratio = cfg.get('test_ratio', 0.1)
    is_parametric = cfg.get('is_parametric', False)

    if not os.path.exists(ori_data_path):
        show_with_end_divider(f"File {ori_data_path} does not exist.")
        raise FileNotFoundError(f"File {ori_data_path} does not exist.")

    df = pd.read_csv(ori_data_path)
    original_prices = df[cfg.get('index')].values
    original_prices = torch.from_numpy(original_prices).float()

    scaler = LogReturnTransformation()
    log_returns, _ = scaler.transform(original_prices)

    if is_parametric:
        return preprocess_parametric(log_returns, original_prices, valid_ratio, test_ratio)
    
    return preprocess_non_parametric(log_returns, original_prices, valid_ratio, test_ratio)