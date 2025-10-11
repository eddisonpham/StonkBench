"""
Path and directory utility functions.

This module provides utilities for path management and directory operations.
"""

import os
from pathlib import Path


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


def ensure_dir(path: Path) -> None:
    """
    Create directory if it does not exist (including parents).
    
    Args:
        path (Path): Path object to create
    """
    path.mkdir(exist_ok=True)
