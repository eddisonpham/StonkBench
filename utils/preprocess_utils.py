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
import mgzip
import pickle
import json
import torch
import numpy as np
import pandas as pd
import random
from statsmodels.tsa.stattools import acf
from torch.utils.data import Dataset, DataLoader
from scipy.signal import argrelextrema

class MinMaxScaler():
    """
    Min-Max normalization scaler implementing the TSGBench normalization step.
    
    This class implements the normalization step from the TSGBench preprocessing pipeline,
    which normalizes the dataset to the range [0, 1] to enhance efficiency and numerical
    stability. This is the final step in the TSGBench pipeline, applied after segmentation,
    shuffling, and train-test splitting.
    
    The normalization formula is: (x - min) / (max - min)
    The inverse transformation is: x * (max - min) + min
    
    This normalization is applied to the segmented sub-matrices {T_r} to ensure:
    - Enhanced efficiency in time series generation models
    - Numerical stability during training and inference
    - Consistent scale across different datasets in the benchmark
    
    Attributes:
        mini (numpy.ndarray): Minimum values for each variable/channel
        range (numpy.ndarray): Range (max - min) for each variable/channel
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
            
        Note:
            Adds small epsilon (1e-7) to prevent division by zero
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
        
        # Convert to PyTorch tensor
        self.data = torch.from_numpy(data).float()
        self.transform = transform
        
        # Create shuffled indices with seed support
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

def show_divider():
    """
    Print a visual divider line for console output formatting.
    
    Prints a line of 20 equal signs to create visual separation
    in console output for better readability.
    """
    print("=" * 20)

def show_with_start_divider(content):
    """
    Display content with a divider line at the start.
    
    Args:
        content (str): Content to display after the divider
    """
    show_divider()
    print(content)

def show_with_end_divider(content):
    """
    Display content with a divider line at the end.
    
    Args:
        content (str): Content to display before the divider
    """
    print(content)
    show_divider()
    print()

def make_sure_path_exist(path):
    """
    Ensure that a directory path exists, creating it if necessary.
    
    This function handles both file paths and directory paths, creating
    the necessary parent directories to ensure the path exists.
    
    Args:
        path (str): File or directory path to ensure exists
    """
    if os.path.isdir(path) and not path.endswith(os.sep):
        dir_path = path
    else:
        # Extract the directory part of the path
        dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

def create_dataloaders(train_data, valid_data, batch_size=32, train_seed=None, valid_seed=None, 
                      num_workers=0, pin_memory=False):
    """
    Create PyTorch DataLoaders for training and validation time series data.
    
    This function creates properly configured DataLoaders that handle time series batching
    with seed support for reproducible training.
    
    Args:
        train_data (numpy.ndarray): Training data with shape (R_train, l, N)
        valid_data (numpy.ndarray): Validation data with shape (R_valid, l, N)
        batch_size (int): Batch size for DataLoaders (default: 32)
        train_seed (int, optional): Seed for training data shuffling (default: None)
        valid_seed (int, optional): Seed for validation data shuffling (default: None)
        num_workers (int): Number of worker processes for data loading (default: 0)
        pin_memory (bool): Whether to pin memory for faster GPU transfer (default: False)
        
    Returns:
        tuple: (train_loader, valid_loader) - PyTorch DataLoader objects
    """
    # Create datasets with seed support
    train_dataset = TimeSeriesDataset(train_data, seed=train_seed)
    valid_dataset = TimeSeriesDataset(valid_data, seed=valid_seed)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, valid_loader

def find_length(data):
    """
    Determine the optimal sequence length l using autocorrelation functions as per TSGBench methodology.
    
    This function implements the sequence length determination step from the TSGBench preprocessing
    pipeline. It employs autocorrelation functions to ensure that each segmented sub-matrix T_r
    encompasses at least one time series period, preserving meaningful temporal structures.
    
    Args:
        data (numpy.ndarray): 1D time series data array
        
    Returns:
        int: Optimal sequence length l (default: 125 if analysis fails or produces invalid results)
    """
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    base = 3
    nobs = len(data)
    nlags = int(min(10 * np.log10(nobs), nobs - 1))
    auto_corr = acf(data, nlags=nlags, fft=True)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125

def sliding_window_view(data, window_size, step=1):
    """
    Implement time series segmentation as per TSGBench preprocessing pipeline.
    
    This function segments long time series T into shorter sub-matrices {T1, T2, T3, ...}
    with specified sequence length l and stride of 1, creating R overlapping sub-matrices {T_r}
    where R = L - l + 1 and each T_r has the same length l.
    
    Args:
        data (numpy.ndarray): 2D array of shape (L, N) where L is length and N is number of variables
        window_size (int): Sequence length l for each sub-matrix T_r
        step (int, optional): Stride between windows. Defaults to 1 (overlapping windows)
        
    Returns:
        numpy.ndarray: 3D array of shape (R, l, N) where R = L - l + 1
    """
    if data.ndim != 2:
        raise ValueError("Input array must be 2D")
    L, C = data.shape  # Length and Channels
    if L < window_size:
        raise ValueError("Window size must be less than or equal to the length of the array")

    # Calculate the number of windows B
    B = L - window_size + 1
    
    # Shape of the output array
    new_shape = (B, window_size, C)
    
    # Calculate strides
    original_strides = data.strides
    new_strides = (original_strides[0],) + original_strides  # (stride for L, stride for W, stride for C)

    # Create the sliding window view
    strided_array = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)
    return strided_array

def save_preprocessed_csv(train_data, valid_data, dataset_name, output_dir):
    """
    Save preprocessed data as CSV files following TSGBench format.
    
    Args:
        train_data (numpy.ndarray): Training data with shape (R_train, l, N)
        valid_data (numpy.ndarray): Validation data with shape (R_valid, l, N)
        dataset_name (str): Name of the dataset
        output_dir (str): Output directory path
        
    Returns:
        tuple: Paths to the created CSV files (train_csv_path, valid_csv_path)
    """
    make_sure_path_exist(output_dir)
    
    # Flatten the 3D data to 2D for CSV format: (R, l*N)
    R_train, l, N = train_data.shape
    R_valid = valid_data.shape[0]
    
    # Reshape to (R, l*N) for CSV format
    train_flat = train_data.reshape(R_train, l * N)
    valid_flat = valid_data.reshape(R_valid, l * N)
    
    # Create column names: time_0_var_0, time_0_var_1, ..., time_l-1_var_N-1
    columns = []
    for t in range(l):
        for n in range(N):
            columns.append(f'time_{t}_var_{n}')
    
    # Create DataFrames
    train_df = pd.DataFrame(train_flat, columns=columns)
    valid_df = pd.DataFrame(valid_flat, columns=columns)
    
    # Add metadata columns
    train_df['split'] = 'train'
    train_df['sequence_id'] = range(R_train)
    valid_df['split'] = 'valid'
    valid_df['sequence_id'] = range(R_valid)
    
    # Save CSV files
    train_csv_path = os.path.join(output_dir, f'{dataset_name}_train_preprocessed.csv')
    valid_csv_path = os.path.join(output_dir, f'{dataset_name}_valid_preprocessed.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    valid_df.to_csv(valid_csv_path, index=False)
    
    print(f"✓ Saved CSV files:")
    print(f"  Training: {train_csv_path} (shape: {train_df.shape})")
    print(f"  Validation: {valid_csv_path} (shape: {valid_df.shape})")
    
    return train_csv_path, valid_csv_path