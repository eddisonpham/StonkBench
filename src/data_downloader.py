"""
Utility script to download and save OHLC stock data using yfinance, for given tickers.
"""

from pathlib import Path
import argparse
import zipfile
import pandas as pd

from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF


def unzip_and_save_csv(zip_file_path, output_dir, cbind=False):
    """
    Unzips a downloaded file and saves the CSV data to a folder based on the zip filename.
    If cbind is True, also reads the extracted CSV into a pandas.DataFrame and returns it.
    Returns:
      - If cbind is False: Path to the first CSV (Path) or None
      - If cbind is True: tuple(Path or None, DataFrame or None)
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
        
        # Find the first CSV file
        csv_files = list(pair_dir.glob('*.csv'))
        if csv_files:
            csv_path = csv_files[0]
            print(f"Successfully extracted CSV file: {csv_path}")
        else:
            print("No CSV files found in the extracted content")
            csv_path = None
        
        # Delete the zip file after extraction
        try:
            zip_file_path.unlink()
            print(f"Deleted zip file: {zip_file_path}")
        except Exception as e:
            print(f"Could not delete zip file: {e}")
        
        if cbind:
            if csv_path is None:
                return (None, None)
            try:
                df = pd.read_csv(csv_path)
                return (csv_path, df)
            except Exception as e:
                print(f"Error reading CSV into DataFrame: {e}")
                return (csv_path, None)
        else:
            return csv_path
    except Exception as e:
        print(f"Error unzipping file: {e}")
        return (None, None) if cbind else None

def main():
    """
    Main function to download and save OHLC data.
    Pass argument to download index/pair and year. Default platform is MetaStock and timeframe is 1 minute.

    python3 src/data_downloader.py --index spxusd nsxusd --year 2022 2023 --cbind True
    """
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).parent.parent
    
    # Download the file
    parser = argparse.ArgumentParser(description="Download OHLC data for a given index and year.")
    parser.add_argument('--index', '-i', nargs='+', default=['spxusd'],
                        help='Index/pair name(s), space-separated (e.g. spxusd spxeur)')
    parser.add_argument('--year', '-y', nargs='+', default=['2023'],
                        help='Year(s) to download, space-separated (e.g. 2021 2022 2023)')
    parser.add_argument('--cbind', '-c', nargs='+', default=True,
                        help='True means to combine all downloaded data into a single DataFrame and save as CSV')
    args = parser.parse_args()

    index = args.index
    years = args.year
    cbind = args.cbind
    years.sort()

    # Iterate over years and indices to download data
    result_df = None
    for i in index:
        index_df = None
        for year in years:
            i = i.lower()
            print(f"Downloading {i} data for year {year}...")
            dl(year=year, month=None, pair=i, platform=P.META_STOCK, time_frame=TF.ONE_MINUTE)
    
            # Find the downloaded zip file in project root directory
            zip_files = sorted(project_root.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
            if not zip_files:
                print("No zip file found in project root directory")
                return

            zip_file = zip_files[0]  # Get the most recently downloaded zip
            print(f"Found zip file: {zip_file}")
            res = unzip_and_save_csv(zip_file, data_dir, cbind=bool(cbind))
            # unzip_and_save_csv returns (csv_path, df) when cbind is True, otherwise a Path or None
            par_df = None
            if isinstance(res, tuple):
                _, par_df = res
            elif isinstance(res, pd.DataFrame):
                par_df = res

            if par_df is not None:
                # reformat par_df
                try:
                    # timestamp = second column
                    ts = par_df.iloc[:, 1].copy()

                    # choose closing price column: prefer 6th column (index 5), otherwise second-to-last
                    close_idx = 5 if par_df.shape[1] > 5 else -2
                    close = par_df.iloc[:, close_idx].copy()
                    close = pd.to_numeric(close, errors='coerce')

                    col_name = i.lower()
                    par_df = pd.DataFrame({'timestamp': ts, col_name: close})
                except Exception as e:
                    print(f"Error reformatting DataFrame: {e}")
                    par_df = pd.DataFrame()  # ensure par_df is a DataFrame even on failure
                # concatenate into index_df
                if index_df is None:
                    index_df = par_df
                else:
                    index_df = pd.concat([index_df, par_df], ignore_index=True, sort=False)
            else:
                print(f"No DataFrame returned for {zip_file}, skipping concatenation.")
        
        if index_df is not None and not index_df.empty:
            # Before merging, ensure index_df has one row per timestamp
            try:
                col_name = i.lower()

                # Normalize timestamp to numeric values and drop rows with invalid timestamps
                index_df['timestamp'] = pd.to_numeric(index_df['timestamp'], errors='coerce')
                index_df = index_df.dropna(subset=['timestamp'])

                # If there are duplicate timestamps (from multiple files), keep the last occurrence
                index_df = index_df.drop_duplicates(subset=['timestamp'], keep='last')

                # Sort by timestamp for deterministic ordering
                index_df = index_df.sort_values('timestamp').reset_index(drop=True)

                if result_df is None:
                    result_df = index_df.copy()
                else:
                    # Normalize result_df timestamps as well
                    result_df['timestamp'] = pd.to_numeric(result_df['timestamp'], errors='coerce')
                    result_df = result_df.dropna(subset=['timestamp'])

                    # Ensure result_df has unique timestamps before join
                    result_df = result_df.drop_duplicates(subset=['timestamp'], keep='last')

                    # Join on timestamp, adding the new column to the right
                    result_df = result_df.set_index('timestamp').join(
                        index_df.set_index('timestamp')[[col_name]],
                        how='outer'
                    ).reset_index()

                    # After join, drop any accidental duplicate timestamp rows
                    result_df = result_df.drop_duplicates(subset=['timestamp'], keep='last')
            except Exception as e:
                print(f"Error merging index_df into result_df: {e}")

    if cbind and result_df is not None and not result_df.empty:
        # Save the final combined DataFrame
        data_processed = Path(__file__).parent.parent / "data" / "processed"
        data_processed.mkdir(parents=True, exist_ok=True)

        indices_str = "_".join([str(it).lower() for it in index])
        years_str = "_".join([str(y) for y in years])
        output_csv = data_processed / f"combined_{indices_str}_{years_str}.csv"
        try:
            result_df.to_csv(output_csv, index=False)
            print(f"Saved combined data to {output_csv}")
        except Exception as e:
            print(f"Error saving combined DataFrame to CSV: {e}")

if __name__ == "__main__":
    main()
