"""
Utility functions for evaluation plotting.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def find_latest_evaluation_folder(results_dir: str = "results") -> str:
    """
    Find the latest evaluation folder based on timestamp in the folder name.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        Path to the latest evaluation folder
        
    Raises:
        FileNotFoundError: If no evaluation folders are found
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory '{results_dir}' not found")
    
    # Find all evaluation folders
    evaluation_folders = []
    for folder in results_path.iterdir():
        if folder.is_dir() and folder.name.startswith("evaluation_"):
            # Extract timestamp from folder name (format: evaluation_YYYYMMDD_HHMMSS)
            match = re.match(r"evaluation_(\d{8}_\d{6})", folder.name)
            if match:
                timestamp_str = match.group(1)
                try:
                    # Parse timestamp
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    evaluation_folders.append((timestamp, folder))
                except ValueError:
                    continue
    
    if not evaluation_folders:
        raise FileNotFoundError("No valid evaluation folders found")
    
    # Sort by timestamp and return the latest
    evaluation_folders.sort(key=lambda x: x[0], reverse=True)
    latest_folder = evaluation_folders[0][1]
    
    return str(latest_folder)


def load_evaluation_data(evaluation_folder: str) -> Dict[str, Any]:
    """
    Load evaluation data from the complete_evaluation.json file.
    
    Args:
        evaluation_folder: Path to the evaluation folder
        
    Returns:
        Dictionary containing all evaluation data
        
    Raises:
        FileNotFoundError: If complete_evaluation.json is not found
        json.JSONDecodeError: If the JSON file is malformed
    """
    evaluation_path = Path(evaluation_folder)
    json_file = evaluation_path / "complete_evaluation.json"
    
    if not json_file.exists():
        raise FileNotFoundError(f"complete_evaluation.json not found in {evaluation_folder}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data


def create_output_directory(output_dir: str = "evaluation_plots") -> str:
    """
    Create the output directory for plots if it doesn't exist.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        Path to the created directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    return str(output_path)
