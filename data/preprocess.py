"""
Dataset preprocessor

This module implements the standardized preprocessing pipeline from the TSGBench paper:
"TSGBench: Time Series Generation Benchmark" - https://arxiv.org/pdf/2309.03755

Standardized Preprocessing Pipeline:
1. Time Series Segmentation: Segment long time series T into shorter sub-matrices {T1, T2, T3, ...}
   with specified sequence length l and stride of 1, creating R overlapping sub-matrices {T_r}
   where R = L - l + 1 and each T_r has the same length l.

2. Sequence Length Determination: Employ autocorrelation functions to determine the optimal
   value of l, ensuring that each T_r encompasses at least one time series period.

3. Data Shuffling: Shuffle the time series to approximate an i.i.d. sample distribution.

4. Train-Test Split: Divide the data into training and testing sets in a 9:1 ratio, allocating
   a larger portion for training and evaluation as is common in TSG methodology.

5. Normalization: Normalize the dataset to the range [0, 1] to enhance efficiency and
   numerical stability, resulting in a dataset shape of (R, l, N).
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import mgzip
import sys
from pathlib import Path
import torch
import random
sys.path.append(str(Path(__file__).parent.parent))
from utils.preprocess_utils import (
    show_with_start_divider, 
    show_with_end_divider, 
    make_sure_path_exist, 
    create_dataloaders,
    find_length,
    sliding_window_view,
    MinMaxScaler, 
    TimeSeriesDataset
)
from data.download import download_goog_history, download_multivariate_time_series_repo

def preprocess_data(cfg):
    """
    Implement the complete TSGBench standardized preprocessing pipeline.
    
    The final dataset has shape (R, l, N) where R = L - l + 1, l is sequence length, N is variables.
    
    Args:
        cfg (dict): Configuration dictionary containing preprocessing parameters:
            - original_data_path (str): Path to the original data file
            - output_ori_path (str): Output directory for preprocessed data (default: './data/ori/')
            - dataset_name (str): Name of the dataset (default: 'dataset')
            - seq_length (int): Manual sequence length l (default: None for auto-detection)
            - valid_ratio (float): Validation set ratio (default: 0.1 for 9:1 split)
            - do_normalization (bool): Whether to normalize to [0, 1] (default: True)
            - seed (int, optional): Random seed for reproducible shuffling (default: None)
    
    Returns:
        tuple: (train_data, valid_data) - Preprocessed datasets with shape (R_train, l, N), (R_valid, l, N)
               Returns None if preprocessing fails
        
    Output Files:
        - {dataset_name}_train.pkl: Compressed training data with shape (R_train, l, N)
        - {dataset_name}_valid.pkl: Compressed validation data with shape (R_valid, l, N)
    """
    show_with_start_divider(f"Data preprocessing with settings:{cfg}")

    # Parse configs
    ori_data_path = cfg.get('original_data_path',None)
    output_ori_path = cfg.get('output_ori_path',r'./data/ori/')
    dataset_name = cfg.get('dataset_name','dataset')
    seq_length = cfg.get('seq_length',None)
    valid_ratio = cfg.get('valid_ratio',0.1)
    do_normalization = cfg.get('do_normalization',True)
    seed = cfg.get('seed', None)

    # Read original data
    if not os.path.exists(ori_data_path):
        show_with_end_divider(f'Original file path {ori_data_path} does not exist.')
        return None
    
    _, ext = os.path.splitext(ori_data_path)
    try:
        if ext in ['.csv']:
            # Handle GOOG stock data CSV
            if 'GOOG' in ori_data_path:
                # Read GOOG CSV with pandas to handle headers properly
                df = pd.read_csv(ori_data_path)
                # Use only numeric columns (Open, High, Low, Close, Volume)
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                ori_data = df[numeric_cols].values
            else:
                # Generic CSV reading
                ori_data = np.loadtxt(ori_data_path, delimiter=",", skiprows=1)
                
        elif ext in ['.txt']:
            # Handle exchange rate data
            if 'exchange_rate' in ori_data_path:
                # Read exchange rate data and convert to CSV format
                ori_data = np.loadtxt(ori_data_path)
                
                # Convert to DataFrame with proper headers
                headers = ['Australia', 'Britain', 'Canada', 'Switzerland', 'China', 'Japan', 'New Zealand', 'Singapore']
                df = pd.DataFrame(ori_data, columns=headers)
                
                # Save as CSV for future reference
                csv_path = ori_data_path.replace('.txt', '.csv')
                df.to_csv(csv_path, index=False)
                print(f"✓ Converted exchange rate data to CSV: {csv_path}")
                
                ori_data = df.values
            else:
                # Generic text file reading
                ori_data = np.loadtxt(ori_data_path)
                
        elif ext in ['.pkl']:
            try:
                with mgzip.open(ori_data_path, 'rb') as f:
                    ori_data = pickle.load(f)
            except (OSError, IOError):
                # If mgzip fails, try reading it as a regular pickle file
                with open(ori_data_path, 'rb') as f:
                    ori_data = pickle.load(f)
        else:
            show_with_end_divider(f"Error: Unsupported file extension: {ext}")
            return None
            
    except Exception as e:
        show_with_end_divider(f"Error: An error occurred during reading data: {e}.")
        return None
    
    # Check and interpolate missing values
    if np.isnan(ori_data).any():
        if not isinstance(ori_data, pd.DataFrame):
            df = pd.DataFrame(ori_data)
        df = df.interpolate(axis=1)
        ori_data = df.to_numpy()

    # Determine the data length
    if seq_length:
        if seq_length>0 and seq_length<=ori_data.shape[0]:
            seq_length = int(seq_length)
        else:
            window_all = []
            for i in range(ori_data.shape[1]):
                window_all.append(find_length(ori_data[:,i]))

            seq_length = int(np.mean(np.array(window_all)))
    
    # Slice the data by sliding window
    # windowed_data = np.lib.stride_tricks.sliding_window_view(ori_data, window_shape=(seq_length, ori_data.shape[1]))
    # windowed_data = np.squeeze(windowed_data, axis=1)
    windowed_data = sliding_window_view(ori_data, seq_length)
    
    # Shuffle with seed support
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    idx = np.random.permutation(len(windowed_data))
    data = windowed_data[idx]
    print('Data shape:', data.shape) 

    train_len = int(data.shape[0] * (1 - valid_ratio))
    train_data = data[:train_len]
    valid_data = data[train_len:]

    if do_normalization:
        scaler = MinMaxScaler()        
        train_data = scaler.fit_transform(train_data)
        valid_data = scaler.transform(valid_data)
    
    # Save preprocessed data
    output_path = os.path.join(output_ori_path,dataset_name)
    make_sure_path_exist(output_path+os.sep)
    with mgzip.open(os.path.join(output_path,f'{dataset_name}_train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with mgzip.open(os.path.join(output_path,f'{dataset_name}_valid.pkl'), 'wb') as f:
        pickle.dump(valid_data, f)

    show_with_end_divider(f'Preprocessing done. Preprocessed files saved to {output_path}.')
    return train_data, valid_data

def load_preprocessed_data(cfg):
    """
    Load previously preprocessed time series data from compressed pickle files.
    
    This function loads training and validation datasets that were previously processed
    by the preprocess_data function. The data is stored in compressed pickle format
    using mgzip for efficient storage and loading.
    
    Args:
        cfg (dict): Configuration dictionary containing loading parameters:
            - dataset_name (str): Name of the dataset to load (default: 'dataset')
            - output_ori_path (str): Directory containing preprocessed data (default: './data/ori/')
    
    Returns:
        tuple: (train_data, valid_data) - Loaded training and validation datasets
               Returns None if loading fails or files don't exist
    """
    show_with_start_divider(f"Load preprocessed data with settings:{cfg}")

    # Parse configs
    dataset_name = cfg.get('dataset_name','dataset')
    output_ori_path = cfg.get('output_ori_path',r'./data/ori/')

    file_path = os.path.join(output_ori_path,dataset_name)
    train_data_path = os.path.join(file_path,f'{dataset_name}_train.pkl')
    valid_data_path = os.path.join(file_path,f'{dataset_name}_valid.pkl')

    # Read preprocessed data
    if not os.path.exists(train_data_path) or not os.path.exists(valid_data_path):
        show_with_end_divider(f'Error: Preprocessed file in {file_path} does not exist.')
        return None
    try:
        with mgzip.open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        with mgzip.open(valid_data_path, 'rb') as f:
            valid_data = pickle.load(f)
    except Exception as e:
        show_with_end_divider(f"Error: An error occurred during reading data: {e}.")
        return None

    show_with_end_divider(f'Preprocessed dataset {dataset_name} loaded.')
    return train_data, valid_data

def create_dataset_from_preprocessed(cfg, batch_size=32, train_seed=None, valid_seed=None, 
                                   num_workers=0, pin_memory=False):
    """
    Create PyTorch datasets and dataloaders from preprocessed data files.
    
    This function loads preprocessed data and creates properly configured PyTorch
    datasets and dataloaders with seed support for reproducible training.
    
    Args:
        cfg (dict): Configuration dictionary containing:
            - dataset_name (str): Name of the dataset to load
            - output_ori_path (str): Directory containing preprocessed data
        batch_size (int): Batch size for DataLoaders (default: 32)
        train_seed (int, optional): Seed for training data shuffling (default: None)
        valid_seed (int, optional): Seed for validation data shuffling (default: None)
        num_workers (int): Number of worker processes for data loading (default: 0)
        pin_memory (bool): Whether to pin memory for faster GPU transfer (default: False)
        
    Returns:
        tuple: (train_loader, valid_loader, train_dataset, valid_dataset)
               Returns None if loading fails
    """
    show_with_start_divider(f"Creating datasets from preprocessed data with settings: {cfg}")
    
    # Load preprocessed data
    train_data, valid_data = load_preprocessed_data(cfg)
    
    if train_data is None or valid_data is None:
        show_with_end_divider("Failed to load preprocessed data")
        return None
    
    # Create datasets and dataloaders
    train_loader, valid_loader = create_dataloaders(
        train_data, valid_data, 
        batch_size=batch_size,
        train_seed=train_seed,
        valid_seed=valid_seed,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Also return the datasets for direct access if needed
    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset
    
    show_with_end_divider(f'Created datasets and dataloaders successfully')
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
    print(f"Batch size: {batch_size}, Train seed: {train_seed}, Valid seed: {valid_seed}")
    
    return train_loader, valid_loader, train_dataset, valid_dataset

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
    
# See example_usage.py for usage examples