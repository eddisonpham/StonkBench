"""
Path and directory utility functions.

This module provides utilities for path management and directory operations.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import mgzip
import io

REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close']

def read_and_format_data(path):
    """
    Converts the Date column from ISO 8601 format (e.g., '2020-12-01 00:00:00-05:00')
    to seconds since the epoch.

    Args:
        path (str): Path to the CSV data file.

    Returns:
        numpy.ndarray: Array of shape (n_samples, n_features) containing the relevant numeric data.
    """
    
    df = pd.read_csv(path)
    assert all(col in df.columns for col in REQUIRED_COLUMNS), \
        f"Input data is missing required columns: {[col for col in REQUIRED_COLUMNS if col not in df.columns]}"
    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce').astype('int64') // 10**9
    return df[REQUIRED_COLUMNS].values