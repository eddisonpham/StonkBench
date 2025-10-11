"""
Data reading utility functions.

This module provides utilities for reading various data formats.
"""

import numpy as np
import pandas as pd
import pickle
import mgzip
import io


def read_csv_data(path):
    """
    Read CSV data, handling special cases like GOOG stock data.
    Returns: numpy.ndarray
    """
    if 'GOOG' in path:
        # Read GOOG CSV with pandas to handle headers properly
        df = pd.read_csv(path)
        # Use only numeric columns (Open, High, Low, Close, Volume)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df[numeric_cols].values
    else:
        # Generic CSV reading
        return np.loadtxt(path, delimiter=",", skiprows=1)

def read_exchange_rate_data(path):
    """
    Reads exchange rate data from a .txt file, adds headers in memory,
    and returns the data as a numpy array without creating new files.
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
    try:
        with mgzip.open(path, 'rb') as f:
            return pickle.load(f)
    except (OSError, IOError):
        with open(path, 'rb') as f:
            return pickle.load(f)
