"""
Utility script for downloading and processing OHLC stock data from histdata using yfinance tickers.
"""

from pathlib import Path
import argparse
import zipfile
from typing import Optional, Tuple, Union
import pandas as pd
from functools import reduce
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF


PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT/ "data" / "raw"
OUT_FOLDER = PROJECT_ROOT / "data" / "processed"


def unzip_and_save_csv(
    zip_file_path: Union[str, Path],
    output_dir: Union[str, Path]
) -> Optional[pd.DataFrame]:
    """Unzip a zip file containing a CSV to a specified directory. Optionally also return the loaded DataFrame.

    Args:
        zip_file_path: Path to ZIP file.
        output_dir: Directory to extract the contents.

    Returns:
        The loaded DataFrame.
        Returns None on failure.
    """
    print(f"Unzipping {zip_file_path} to {output_dir}")
    zip_file_path, output_dir = Path(zip_file_path), Path(output_dir)

    pair_name = zip_file_path.stem.split('_')[2] if len(zip_file_path.stem.split('_')) > 2 else 'data'
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

    indices = args.index
    years = args.year
    years.sort()

    all_columns = ["timestamp"] + [i.lower() for i in indices]
    combined_df_list = []

    for i in indices:
        i_lower = i.lower()
        temp_df_list = []

        for year in years:
            print(f"Downloading {i_lower} {year}")
            dl(year=year, month=None, pair=i_lower,
               platform=P.META_STOCK, time_frame=TF.ONE_MINUTE)

            candidate_zips = [
                p for p in PROJECT_ROOT.glob('*.zip')
                if year in p.name and i_lower in p.name.lower()
            ]
            candidate_zips = sorted(
                candidate_zips,
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if not candidate_zips:
                print(f"No zip found for {i_lower} {year}.")
                continue

            zip_path = candidate_zips[0]
            par_df = unzip_and_save_csv(zip_path, DATA_FOLDER)

            if par_df is None:
                print(f"Failed to unzip and load CSV from {zip_path}. Skipping.")
                continue

            ts_col = 1
            val_col = 5 if par_df.shape[1] > 5 else max(0, par_df.shape[1] - 2)

            temp_col_df = pd.DataFrame({
                'timestamp': pd.to_numeric(par_df.iloc[:, ts_col], errors='coerce'),
                i_lower: pd.to_numeric(par_df.iloc[:, val_col], errors='coerce')
            })

            temp_col_df = (
                temp_col_df
                .dropna(subset=['timestamp'])
                .sort_values('timestamp', ignore_index=True)
            )

            temp_df_list.append(temp_col_df)

        if not temp_df_list:
            print(f"No data collected for index {i_lower}.")
            continue

        index_df = (
            pd.concat(temp_df_list, ignore_index=True)
            .drop_duplicates('timestamp', keep='last')
            .sort_values('timestamp')
            .reset_index(drop=True)
        )

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
    safe_name = "_".join([col for col in all_columns if col != 'timestamp']).replace(" ", "_")
    output_csv = OUT_FOLDER / f"combined_{safe_name}.csv"
    combined_df.to_csv(output_csv, index=False)
    print(f"Saved combined CSV to {output_csv}")

if __name__ == "__main__":
    main()
