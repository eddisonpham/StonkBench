"""
Utility script for downloading and processing OHLC stock data from histdata using yfinance tickers.
"""

from pathlib import Path
import argparse
import zipfile
from typing import Optional, Union, List
import pandas as pd
from functools import reduce
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF
import shutil


PROJECT_ROOT = Path(__file__).parent.parent
if Path("/data").exists():
    DATA_FOLDER = Path("/data") / "raw"
    OUT_FOLDER = Path("/data") / "processed"
else:
    DATA_FOLDER = PROJECT_ROOT / "data" / "raw"
    OUT_FOLDER = PROJECT_ROOT / "data" / "processed"


def unzip_and_save_csv(
    zip_file_path: Union[str, Path],
    output_dir: Union[str, Path],
    year: Optional[str]
) -> Optional[pd.DataFrame]:
    """Unzip a zip file containing a CSV to a specified directory. Optionally also return the loaded DataFrame."""

    print(f"Unzipping {zip_file_path} to {output_dir}")

    zip_file_path, output_dir = Path(zip_file_path), Path(output_dir)
    pair_name = zip_file_path.stem.split('_')[2] if len(zip_file_path.stem.split('_')) > 2 else 'data'
    pair_dir = output_dir / pair_name / year
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
    """Find the most recently downloaded zip file for a given index and year."""
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
    """Download and process data for a single index-year combination."""

    print(f"Downloading {index} {year}")
    
    dl(year=year, pair=index, platform=P.META_STOCK, time_frame=TF.ONE_MINUTE)
    zip_path = find_zip_file(index, year)

    if zip_path is None:
        print(f"[ERROR] No zip found for {index} {year}.")
        return None

    par_df = unzip_and_save_csv(zip_path, DATA_FOLDER, year=year)
    if par_df is None:
        print(f"[ERROR] Failed to unzip and load CSV from {zip_path}. Skipping.")
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
    """
    Combine multiple dataframes for the same index, removing duplicates.
    The result is only for a single index (multiple years).
    """
    if not df_list:
        return pd.DataFrame()
    return (
        pd.concat(df_list, ignore_index=True)
        .drop_duplicates('timestamp', keep='last')
        .sort_values('timestamp')
        .reset_index(drop=True)
    )


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing OHLC values for all columns except 'timestamp'."""
    if df.empty:
        return df
    columns_to_impute = [col for col in df.columns if col != "timestamp"]
    if not columns_to_impute:
        return df
    df = df.set_index("timestamp")
    df[columns_to_impute] = df[columns_to_impute].ffill()
    return df.reset_index()


def clear_directories(*dirs):
    """Remove the contents of all given directories, if they exist, but not the directories themselves."""
    for d in dirs:
        d = Path(d)
        if d.exists() and d.is_dir():
            for item in d.iterdir():
                if item.is_file() or item.is_symlink():
                    try:
                        item.unlink()
                    except Exception as e:
                        print(f"Warning: Could not delete file: {item} ({e})")
                elif item.is_dir():
                    try:
                        shutil.rmtree(item)
                    except Exception as e:
                        print(f"Warning: Could not delete directory: {item} ({e})")


def clip_to_common_timestamps(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Restrict all dataframes to their common timestamp range (intersection).
    """
    min_ts = max(df['timestamp'].min() for df in dfs)
    max_ts = min(df['timestamp'].max() for df in dfs)
    return [df[(df['timestamp'] >= min_ts) & (df['timestamp'] <= max_ts)] if not df.empty else df for df in dfs]


def main():
    """Main entrypoint for downloading, extracting, and processing OHLC data."""

    # Clearing the folders before running, so output overwrites are expected
    clear_directories(DATA_FOLDER, OUT_FOLDER)

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Download OHLC data.")
    parser.add_argument('--index', '-i', nargs='+', default=['spxusd', 'eurusd'])
    parser.add_argument('--year', '-y', nargs='+', default=['2023', '2024'])
    args = parser.parse_args()

    indices = [i.lower() for i in args.index]
    years = sorted(args.year)

    # For each index/asset: combine all years for that asset into one dataframe
    index_dfs = []
    for index in indices:
        yearly_dfs = [download_and_process_year(index, year) for year in years]
        yearly_dfs = [df for df in yearly_dfs if df is not None and not df.empty]
        if not yearly_dfs:
            continue
        combined_index_df = combine_index_dataframes(yearly_dfs)
        if not combined_index_df.empty:
            index_dfs.append(combined_index_df)
        else:
            print(f"[WARN] No data for index {index}")

    if not index_dfs:
        print("[ERROR] No dataframes to combine.")
        return

    index_dfs_clipped = clip_to_common_timestamps(index_dfs)
    if not index_dfs_clipped or any(df.empty for df in index_dfs_clipped):
        print("[ERROR] At least one index has no overlap in timestamp range.")
        return

    # Merge all on timestamp (outer join would introduce NaNs; inner join restricts to intersection)
    df_merged = index_dfs_clipped[0]
    for next_df in index_dfs_clipped[1:]:
        # Only keep 'timestamp' + the asset column
        value_cols = [col for col in next_df.columns if col != "timestamp"]
        df_merged = pd.merge(df_merged, next_df, on="timestamp", how="inner")

    # Defensive: Coerce timestamp to numeric, re-sort, remove dupes
    df_merged['timestamp'] = pd.to_numeric(df_merged['timestamp'], errors='coerce')
    df_merged = (
        df_merged
        .dropna(subset=['timestamp'])
        .drop_duplicates('timestamp', keep='last')
        .sort_values('timestamp')
        .reset_index(drop=True)
    )

    df_merged = impute_missing_values(df_merged)

    safe_name = "_".join(indices).replace(" ", "_")
    output_csv = OUT_FOLDER / f"combined_data.csv"

    df_merged.to_csv(output_csv, index=False)
    print(f"Saved combined CSV to {output_csv}")


if __name__ == "__main__":
    main()
