"""
Utility script to download and save OHLC stock data using yfinance, for given tickers.
"""

from pathlib import Path
import argparse
import zipfile
import pandas as pd

from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF


def unzip_and_save_csv(zip_file_path, output_dir):
    """
    Unzips a downloaded file and saves the CSV data to a folder based on the zip filename.
    """
    zip_file_path = Path(zip_file_path)
    output_dir = Path(output_dir)
    
    # Extract pair name from zip filename (e.g., "DAT_ASCII_SPXUSD_T_201906.zip" -> "SPXUSD")
    zip_name = zip_file_path.stem  # Remove .zip extension
    parts = zip_name.split('_')
    pair_name = parts[2] if len(parts) > 2 else 'data'
    
    pair_dir = output_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all files to pair-specific folder
            zip_ref.extractall(pair_dir)
        
        # Find and report the first CSV file
        csv_files = list(pair_dir.glob('*.csv'))
        if csv_files:
            print(f"Successfully extracted CSV file: {csv_files[0]}")
        else:
            print("No CSV files found in the extracted content")
        
        # Delete the zip file after extraction
        zip_file_path.unlink()
        print(f"Deleted zip file: {zip_file_path}")
        
        return csv_files[0] if csv_files else None
    except Exception as e:
        print(f"Error unzipping file: {e}")
        return None

def main():
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    project_root = Path(__file__).parent.parent
    
    # Download the file
    dl(year='2023', month=None, pair='spxusd', platform=P.META_STOCK, time_frame=TF.ONE_MINUTE)
    
    # Find the downloaded zip file in project root directory
    zip_files = sorted(project_root.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
    if zip_files:
        zip_file = zip_files[0]  # Get the most recently downloaded zip
        print(f"Found zip file: {zip_file}")
        unzip_and_save_csv(zip_file, data_dir)
    else:
        print("No zip file found in project root directory")


if __name__ == "__main__":
    main()
