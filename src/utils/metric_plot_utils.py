"""
Utility functions for evaluation plotting.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


def find_sequence_folders(results_dir: str = "results") -> List[str]:
    """
    Find all sequence length folders (seq_*) in the results directory.
    Returns a sorted list of folder paths.
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory '{results_dir}' not found")
    
    seq_folders = []
    for folder in results_path.iterdir():
        if folder.is_dir() and folder.name.startswith("seq_"):
            seq_folders.append(str(folder))
    
    if not seq_folders:
        raise FileNotFoundError("No valid sequence folders (seq_*) found")
    
    # Sort by sequence length
    def extract_seq_length(path_str):
        match = re.match(r".*seq_(\d+)", path_str)
        return int(match.group(1)) if match else 0
    
    seq_folders.sort(key=extract_seq_length)
    return seq_folders


def load_evaluation_data(seq_folder: str) -> Dict[str, Any]:
    """
    Load evaluation data from all model metrics.json files in a sequence folder.
    Returns a dictionary with structure: {model_name: metrics_dict}
    """
    seq_path = Path(seq_folder)
    
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence folder '{seq_folder}' not found")
    
    data = {}
    for model_folder in seq_path.iterdir():
        if model_folder.is_dir():
            metrics_file = model_folder / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    model_data = json.load(f)
                    data[model_folder.name] = model_data
    
    if not data:
        raise FileNotFoundError(f"No metrics.json files found in {seq_folder}")
    
    return data


def load_all_sequence_data(results_dir: str = "results", exclude_seq_52: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Load evaluation data from all sequence folders.
    Returns a dictionary with structure: {seq_folder_name: {model_name: metrics_dict}}
    """
    seq_folders = find_sequence_folders(results_dir)
    all_data = {}
    
    for seq_folder in seq_folders:
        seq_name = Path(seq_folder).name
        if exclude_seq_52 and seq_name == "seq_52":
            continue
        all_data[seq_name] = load_evaluation_data(seq_folder)
    
    return all_data


def create_output_directory(output_dir: str = "evaluation_plots") -> str:
    """
    Create the output directory for plots if it doesn't exist.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    return str(output_path)
