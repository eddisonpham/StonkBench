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
from src.utils.dat_io_utils import read_csv_data 

from src.utils.display_utils import (
    show_with_start_divider,
    show_with_end_divider
)


def preprocess_data(cfg):
    """
    Implements the complete TSGBench standardized preprocessing pipeline for both parametric and non-parametric model types.

    For non-parametric models:
        - Outputs: Dataset of shape (R, l, N), where R = L - l + 1, l is the sub-sequence length, and N is the number of variables.
        - Process: Data is segmented into overlapping sub-sequences of length l with a stride of 1, optionally normalized.

    For parametric models:
        - Outputs: The original (or interpolated) data of shape (l, N) (no segmentation; l = original sequence length).
        - Process: No sub-sequencing is performed; only missing values are imputed if present.

    Args:
        cfg (dict): Configuration dictionary containing preprocessing parameters:
            - original_data_path (str): Path to the original data file.
            - seq_length (int, optional): Sub-sequence length l for segmentation (default: None; automatically determined if not supplied).
            - valid_ratio (float, optional): Fraction of data for validation (default: 0.1, corresponds to 9:1 train-validation split).
            - do_normalization (bool, optional): Whether to normalize data to [0, 1] (default: True).
            - is_parametric (bool, optional): If True, run parametric preprocessing (no segmentation); if False, run non-parametric (default: False).
            - seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple or None: 
            - If successful: (train_data, valid_data)
                * For non-parametric (np.ndarray): arrays of shape (R_train, l, N) and (R_valid, l, N)
                * For parametric (torch.Tensor): arrays of shape (l, N) for both train and validation splits (if splitting applied)
            - If preprocessing fails: None
    """
    show_with_start_divider(f"Data preprocessing with settings:{cfg}")

    # Extract preprocessing configurations
    ori_data_path = cfg.get('original_data_path',None)
    seq_length = cfg.get('seq_length',None)
    valid_ratio = cfg.get('valid_ratio',0.1)
    do_normalization = cfg.get('do_normalization',True)
    is_parametric = cfg.get('is_parametric', False)
    seed = cfg.get('seed', None)

    # Data path exists?
    if not os.path.exists(ori_data_path):
        curr_dir = os.getcwd()
        print(f"Current working directory: {curr_dir}")
        show_with_end_divider(f'Original file path {ori_data_path} does not exist.')
        return None
    
    # Determine extension to read from
    _, ext = os.path.splitext(ori_data_path)
    try:
        if ext == '.csv':
            # Read the CSV data and convert Date (ISO 8601) to seconds since epoch
            ori_data = read_csv_data(ori_data_path)
        else:
            show_with_end_divider(f"Error: Unsupported file extension: {ext}")
            return None
    except Exception as e:
        show_with_end_divider(f"Error: An error occurred during reading data: {e}.")
        return None

    # Impute missing value with interpolation
    if np.isnan(ori_data).any():
        if not isinstance(ori_data, pd.DataFrame):
            df = pd.DataFrame(ori_data)
        df = df.interpolate(axis=1)
        ori_data = df.to_numpy()

    # Early return entire preprocessed dataset if parametric
    if is_parametric:
        ori_data = torch.from_numpy(ori_data)
        print('Data shape:', tuple(ori_data.size()))
        train_len = int(ori_data.shape[0] * (1 - valid_ratio))
        train_data = ori_data[:train_len]
        valid_data = ori_data[train_len:]
        return train_data, valid_data
    else:
        if seq_length is None:
            window_all = np.apply_along_axis(find_length, axis=0, arr=ori_data[:, 1:])
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

            # All but the timestamp channel
            train_feats = train_data[:, :, 1:]
            valid_feats = valid_data[:, :, 1:]

            # Fit multivariate MinMaxScaler on all but timestamp channel
            train_feats = scaler.fit_transform(train_feats)
            valid_feats = scaler.transform(valid_feats)

            # Concatenate channelwise
            train_data = np.concatenate([train_data[:, :, [0]], train_feats], axis=-1)
            valid_data = np.concatenate([valid_data[:, :, [0]], valid_feats], axis=-1)

    show_with_end_divider(f'Preprocessing done.')
    return train_data, valid_data
