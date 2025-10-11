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

4. Train-Test Split: Divide the data into training and testing sets in a 90-10 ratio, allocating
   a larger portion for training and evaluation as is common in TSG methodology.

5. Normalization: Normalize the dataset to the range [0, 1] to enhance efficiency and
   numerical stability, resulting in a dataset shape of (R, l, N).
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import torch
import random

from src.preprocessing.transformers import (
    MinMaxScaler, 
    TimeSeriesDataset,
    create_dataloaders,
    find_length,
    sliding_window_view
)
from src.utils.path_utils import make_sure_path_exist
from src.utils.dat_io_utils import (
    save_pickle_file,
    load_pickle_file,
    read_csv_data 
)
from src.utils.display_utils import (
    show_with_start_divider,
    show_with_end_divider
)


def preprocess_data(cfg):
    """
    Implement the complete TSGBench standardized preprocessing pipeline.
    
    The final dataset has shape (R, l, N) where R = L - l + 1, l is sequence length, N is variables.
    
    Args:
        cfg (dict): Configuration dictionary containing preprocessing parameters:
            - original_data_path (str): Path to the original data file
            - output_ori_path (str): Output directory for preprocessed data (default: './data/preprocessed/')
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

    ori_data_path = cfg.get('original_data_path',None)
    output_ori_path = cfg.get('output_ori_path',r'./data/preprocessed/')
    dataset_name = cfg.get('dataset_name','dataset')
    seq_length = cfg.get('seq_length',None)
    valid_ratio = cfg.get('valid_ratio',0.1)
    do_normalization = cfg.get('do_normalization',True)
    seed = cfg.get('seed', None)

    if not os.path.exists(ori_data_path):
        curr_dir = os.getcwd()
        print(f"Current working directory: {curr_dir}")
        show_with_end_divider(f'Original file path {ori_data_path} does not exist.')
        return None
    
    _, ext = os.path.splitext(ori_data_path)
    try:
        if ext == '.csv':
            ori_data = read_csv_data(ori_data_path)
        elif ext == '.pkl':
            ori_data = load_pickle_file(ori_data_path)
        else:
            show_with_end_divider(f"Error: Unsupported file extension: {ext}")
            return None
    except Exception as e:
        show_with_end_divider(f"Error: An error occurred during reading data: {e}.")
        return None

    if np.isnan(ori_data).any():
        if not isinstance(ori_data, pd.DataFrame):
            df = pd.DataFrame(ori_data)
        df = df.interpolate(axis=1)
        ori_data = df.to_numpy()
    
    if seq_length is None:
        window_all = np.apply_along_axis(find_length, axis=0, arr=ori_data)
        seq_length = int(np.mean(window_all))

    windowed_data = sliding_window_view(ori_data, seq_length)
    
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
    
    output_path = os.path.join(output_ori_path,dataset_name)
    make_sure_path_exist(output_path+os.sep)
    save_pickle_file(train_data, os.path.join(output_path,f'{dataset_name}_train.pkl'))
    save_pickle_file(valid_data, os.path.join(output_path,f'{dataset_name}_valid.pkl'))

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
            - output_ori_path (str): Directory containing preprocessed data (default: './data/preprocessed/')
    
    Returns:
        tuple: (train_data, valid_data) - Loaded training and validation datasets
               Returns None if loading fails or files don't exist
    """
    show_with_start_divider(f"Load preprocessed data with settings:{cfg}")

    dataset_name = cfg.get('dataset_name','dataset')
    output_ori_path = cfg.get('output_ori_path',r'./data/preprocessed/')

    file_path = os.path.join(output_ori_path,dataset_name)
    train_data_path = os.path.join(file_path,f'{dataset_name}_train.pkl')
    valid_data_path = os.path.join(file_path,f'{dataset_name}_valid.pkl')

    if not os.path.exists(train_data_path) or not os.path.exists(valid_data_path):
        show_with_end_divider(f'Error: Preprocessed file in {file_path} does not exist.')
        return None
    try:
        train_data = load_pickle_file(train_data_path)
        valid_data = load_pickle_file(valid_data_path)
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
    
    train_data, valid_data = load_preprocessed_data(cfg)
    
    if train_data is None or valid_data is None:
        show_with_end_divider("Failed to load preprocessed data")
        return None

    train_loader, valid_loader = create_dataloaders(
        train_data, valid_data, 
        batch_size=batch_size,
        train_seed=train_seed,
        valid_seed=valid_seed,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset
    
    show_with_end_divider(f'Created datasets and dataloaders successfully')
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
    print(f"Batch size: {batch_size}, Train seed: {train_seed}, Valid seed: {valid_seed}")
    
    return train_loader, valid_loader, train_dataset, valid_dataset
