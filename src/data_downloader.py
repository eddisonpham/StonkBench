"""
Utility script for downloading and processing OHLC stock data from histdata using yfinance tickers.
"""

from pathlib import Path
import argparse
import zipfile
from itertools import product
from collections import defaultdict
from typing import Optional, Tuple, Union, List
import pandas as pd
from functools import reduce
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF


PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT/ "data" / "raw"
OUT_FOLDER = PROJECT_ROOT / "data" / "processed"


def unzip_and_save_csv(
    zip_file_path: Union[str, Path],
    output_dir: Union[str, Path],
    year: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Unzip a zip file containing a CSV to a specified directory. Optionally also return the loaded DataFrame.

    Args:
        zip_file_path: Path to ZIP file.
        output_dir: Directory to extract the contents.
        year: Optional year to include in the extraction path to avoid overwriting.

    Returns:
        The loaded DataFrame.
        Returns None on failure.
    """
    print(f"Unzipping {zip_file_path} to {output_dir}")
    zip_file_path, output_dir = Path(zip_file_path), Path(output_dir)

    pair_name = zip_file_path.stem.split('_')[2] if len(zip_file_path.stem.split('_')) > 2 else 'data'
    if year:
        pair_dir = output_dir / pair_name / year
    else:
        pair_dir = output_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            z.extractall(pair_dir)

        csv_files = list(pair_dir.glob('*.csv'))
        csv_path = csv_files[0] if csv_files else None

        zip_file_path.unlink()
        
        if csv_path is None:
            return None

        df = pd.read_csv(csv_path)
        return df

    except Exception as e:
        print(f"Unzip error: {e}")
        return None


def find_zip_file(index: str, year: str) -> Optional[Path]:
    """Find the most recently downloaded zip file for a given index and year.
    
    Args:
        index: Index name (lowercase).
        year: Year as string.
        
    Returns:
        Path to the zip file, or None if not found.
    """
    candidate_zips = [
        p for p in PROJECT_ROOT.glob('*.zip')
        if _matches_zip_file(p, index, year)
    ]
    
    if not candidate_zips:
        return None
    
    return max(candidate_zips, key=lambda p: p.stat().st_mtime)


def _matches_zip_file(path: Path, index: str, year: str) -> bool:
    """Check if a zip file matches the given index and year."""
    name_lower = path.name.lower()
    index_lower = index.lower()
    year_str = str(year)
    
    year_matches = (
        f'_{year_str}_' in name_lower or
        name_lower.endswith(f'_{year_str}.zip') or
        name_lower.startswith(f'{year_str}_')
    )
    
    return year_matches and index_lower in name_lower


def download_and_process_year(index: str, year: str) -> Optional[pd.DataFrame]:
    """Download and process data for a single index-year combination.
    
    Args:
        index: Index name (lowercase).
        year: Year as string.
        
    Returns:
        DataFrame with timestamp and index column, or None on failure.
    """
    print(f"Downloading {index} {year}")
    dl(year=year, month=None, pair=index, platform=P.META_STOCK, time_frame=TF.ONE_MINUTE)
    
    zip_path = find_zip_file(index, year)
    if zip_path is None:
        print(f"No zip found for {index} {year}.")
        return None
    
    par_df = unzip_and_save_csv(zip_path, DATA_FOLDER, year=year)
    if par_df is None:
        print(f"Failed to unzip and load CSV from {zip_path}. Skipping.")
        return None
    
    ts_col = 1
    val_col = 5 if par_df.shape[1] > 5 else max(0, par_df.shape[1] - 2)
    
    temp_col_df = pd.DataFrame({
        'timestamp': pd.to_numeric(par_df.iloc[:, ts_col], errors='coerce'),
        index: pd.to_numeric(par_df.iloc[:, val_col], errors='coerce')
    })
    
    return (
        temp_col_df
        .dropna(subset=['timestamp'])
        .sort_values('timestamp', ignore_index=True)
    )


def combine_index_dataframes(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple dataframes for the same index, removing duplicates."""
    if not df_list:
        return pd.DataFrame()
    
    return (
        pd.concat(df_list, ignore_index=True)
        .drop_duplicates('timestamp', keep='last')
        .sort_values('timestamp')
        .reset_index(drop=True)
    )

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill missing OHLC values for all columns except 'timestamp'.
    """
    if df.empty:
        return df

    columns_to_impute = [col for col in df.columns if col != "timestamp"]
    if not columns_to_impute:
        return df

    df = df.set_index("timestamp")
    df[columns_to_impute] = df[columns_to_impute].ffill()
    return df.reset_index()


def main() -> None:
    """Main entrypoint for downloading, extracting, and processing OHLC data."""
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Download OHLC data.")
    parser.add_argument('--index', '-i', nargs='+', default=['spxusd'])
    parser.add_argument('--year', '-y', nargs='+', default=['2023'])
    args = parser.parse_args()

    indices = [i.lower() for i in args.index]
    years = sorted(args.year)
    
    index_dataframes = defaultdict(list)
    for index, year in product(indices, years):
        df = download_and_process_year(index, year)
        if df is not None:
            index_dataframes[index].append(df)
    
    combined_df_list = []
    for index, df_list in index_dataframes.items():
        if not df_list:
            print(f"No data collected for index {index}.")
            continue
        index_df = combine_index_dataframes(df_list)
        combined_df_list.append(index_df)
    
    if not combined_df_list:
        print("No data to save.")
        return

    combined_df = reduce(
        lambda left, right: pd.merge(left, right, on='timestamp', how='outer'),
        combined_df_list
    )

    combined_df['timestamp'] = pd.to_numeric(combined_df['timestamp'], errors='coerce')
    combined_df = (
        combined_df
        .dropna(subset=['timestamp'])
        .drop_duplicates('timestamp', keep='last')
        .sort_values('timestamp')
        .reset_index(drop=True)
    )

    combined_df = impute_missing_values(combined_df)

    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    safe_name = "_".join(indices).replace(" ", "_")
    output_csv = OUT_FOLDER / f"combined_{safe_name}.csv"
    combined_df.to_csv(output_csv, index=False)
    print(f"Saved combined CSV to {output_csv}")

if __name__ == "__main__":
    main()
