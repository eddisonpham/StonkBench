"""
Data input/output utility functions.

This module provides utilities for reading and writing various data formats.
"""

import numpy as np
import pandas as pd
import pickle
import mgzip
import io


import pandas as pd
import pickle
import mgzip

REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

def read_csv_data(path):
    """
    Read CSV file containing stock or generic time series data.

    If the file corresponds to GOOG stock data (identified by 'GOOG' in the filename),
    it reads the file with pandas, expecting the following columns:
        - Open: Open price
        - High: Highest price
        - Low: Lowest price
        - Close: Closing price
        - Volume: Trade volume

    Args:
        path (str): Path to the CSV data file.

    Returns:
        numpy.ndarray: Array of shape (n_samples, n_features) containing the relevant numeric data.
    """
    df = pd.read_csv(path)
    if all(col in df.columns for col in REQUIRED_COLUMNS):
        return df[REQUIRED_COLUMNS].values


def read_exchange_rate_data(path):
    """
    Reads exchange rate data from a .txt file, adds headers in memory,
    and returns the data as a numpy array without creating new files.
    
    Args:
        path (str): Path to the exchange rate data file
        
    Returns:
        numpy.ndarray: Exchange rate data
    """
    headers = ['Australia', 'Britain', 'Canada', 'Switzerland', 'China', 'Japan', 'New Zealand', 'Singapore']
    # Read lines from the file
    with open(path, 'r') as f:
        lines = f.readlines()
    # Prepend header to the lines and read into pandas using StringIO
    csv_content = ','.join(headers) + '\n' + ''.join(lines)
    df = pd.read_csv(io.StringIO(csv_content))
    return df.values


def load_pickle_file(path):
    """
    Load data from a pickle file, handling both compressed and uncompressed formats.
    
    Args:
        path (str): Path to the pickle file
        
    Returns:
        Any: Data loaded from the pickle file
        
    Raises:
        Exception: If file cannot be loaded
    """
    try:
        with mgzip.open(path, 'rb') as f:
            return pickle.load(f)
    except (OSError, IOError):
        with open(path, 'rb') as f:
            return pickle.load(f)


def save_pickle_file(data, path, compress=True):
    """
    Save data to a pickle file, with optional compression.
    
    Args:
        data: Data to save
        path (str): Path to save the pickle file
        compress (bool): Whether to use compression (default: True)
    """
    if compress:
        with mgzip.open(path, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
