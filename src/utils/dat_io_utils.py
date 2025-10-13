"""
Data input/output utility functions.

This module provides utilities for reading and writing various data formats.
"""

import numpy as np
import pandas as pd
import pickle
import mgzip
import io

REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

def read_csv_data(path):
    """
    Read CSV file containing stock or generic time series data.

    Converts the Date column from ISO 8601 format (e.g., '2020-12-01 00:00:00-05:00')
    to seconds since the epoch.

    Args:
        path (str): Path to the CSV data file.

    Returns:
        numpy.ndarray: Array of shape (n_samples, n_features) containing the relevant numeric data.
    """
    df = pd.read_csv(path)
    if all(col in df.columns for col in REQUIRED_COLUMNS):
        df = df.copy()
        # Convert Date column to seconds since epoch (UTC), robust to timezones and mixed zones
        df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
        # Fill any invalid or missing datetimes with NaT
        mask_nan = df['Date'].isna()
        if mask_nan.any():
            raise ValueError(f"Unable to parse following date rows: {df.loc[mask_nan, 'Date']}")
        # Convert to seconds since epoch (UTC)
        df['Date'] = df['Date'].view('int64') // 10**9
        return df[REQUIRED_COLUMNS].values

