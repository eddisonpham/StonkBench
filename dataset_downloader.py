"""
Dataset Downloader

This script provides three independent functions to download datasets into the
`Datasets/` directory:

1) download_goog_recent_history: Recent GOOG stock price history (CSV)
2) download_goog_long_history: Long/max range GOOG stock price history (CSV)
3) download_multivariate_time_series_repo: Time-series datasets repo (zip -> extracted)

Usage examples (PowerShell / bash):
  python dataset_downloader.py --all
  python dataset_downloader.py --goog-recent
  python dataset_downloader.py --goog-long
  python dataset_downloader.py --mts-repo

The outputs are saved under `Datasets/` in their respective subfolders.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import yfinance as yf
import io
import zipfile
import requests
import gzip


def _ensure_dir(path: Path) -> None:
    """Create directory if it does not exist (including parents)."""
    path.mkdir(parents=True, exist_ok=True)


def download_goog_recent_history(output_root: Path = Path("Datasets"),
                                 ticker: str = "GOOG",
                                 period: str = "5y",
                                 interval: str = "1d") -> Path:
    """Download recent history for a Google (GOOG) ticker to CSV.

    - Uses yfinance for robust data access.
    - Default period is 5 years of daily candles.

    Returns the CSV file path.
    """
    output_dir = output_root / f"{ticker}_recent"
    _ensure_dir(output_dir)
    output_csv = output_dir / f"{ticker}.csv"

    # Fetch data
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period=period, interval=interval, auto_adjust=False)

    if df is None or df.empty:
        raise RuntimeError("No data returned from Yahoo Finance for the requested period.")

    # Persist
    df.to_csv(output_csv, index=True)
    return output_csv


def download_goog_long_history(output_root: Path = Path("Datasets"),
                               ticker: str = "GOOG",
                               start: str | None = None,
                               end: str | None = None,
                               interval: str = "1d") -> Path:
    """Download long (max range) history for Google (GOOG) to CSV.

    - If start/end are not provided, fetches the maximum available range.
    - Uses yfinance. Output saved in `Datasets/GOOG_long/GOOG.csv`.

    Returns the CSV file path.
    """
    output_dir = output_root / f"{ticker}_long"
    _ensure_dir(output_dir)
    output_csv = output_dir / f"{ticker}.csv"

    ticker_obj = yf.Ticker(ticker)
    if start is None and end is None:
        df = ticker_obj.history(period="max", interval=interval, auto_adjust=False)
    else:
        df = ticker_obj.history(start=start, end=end, interval=interval, auto_adjust=False)

    if df is None or df.empty:
        raise RuntimeError("No data returned from Yahoo Finance for the requested span.")

    df.to_csv(output_csv, index=True)
    return output_csv


def download_multivariate_time_series_repo(output_root: Path = Path("Datasets")) -> Path:
    """Download and extract only the exchange_rate folder from the multivariate time-series datasets repository.

    Source: https://github.com/laiguokun/multivariate-time-series-data

    The zip is fetched from the default branch and only the exchange_rate folder is extracted.
    The exchange_rate.txt.gz file is also extracted to exchange_rate.txt.

    Returns the extraction directory path.
    """
    # Direct link to the repository zip of the default branch
    repo_zip_url = (
        "https://github.com/laiguokun/multivariate-time-series-data/archive/refs/heads/master.zip"
    )

    extraction_dir = output_root / "multivariate-time-series-data"
    _ensure_dir(extraction_dir)

    # Download zip into memory (sufficient for a small repo); could stream to disk if desired
    resp = requests.get(repo_zip_url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download repo zip. HTTP {resp.status_code}")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Extract only the exchange_rate folder
        for file_info in zf.filelist:
            if file_info.filename.startswith("multivariate-time-series-data-master/exchange_rate/"):
                # Remove the top-level folder prefix and extract to target directory
                target_path = file_info.filename.replace("multivariate-time-series-data-master/", "")
                file_info.filename = target_path
                zf.extract(file_info, extraction_dir)

    # Extract the exchange_rate.txt.gz file
    gz_file_path = extraction_dir / "exchange_rate" / "exchange_rate.txt.gz"
    txt_file_path = extraction_dir / "exchange_rate" / "exchange_rate.txt"
    
    if gz_file_path.exists():
        with gzip.open(gz_file_path, 'rb') as gz_file:
            with open(txt_file_path, 'wb') as txt_file:
                txt_file.write(gz_file.read())
        print(f"Extracted {gz_file_path} to {txt_file_path}")

    return extraction_dir


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets into the Datasets/ directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--goog-recent", action="store_true", help="Download GOOG recent history")
    parser.add_argument("--goog-long", action="store_true", help="Download GOOG long (max) history")
    parser.add_argument("--mts-repo", action="store_true", help="Download multivariate time-series repo")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # Ensure root output directory exists
    datasets_root = Path("Datasets")
    _ensure_dir(datasets_root)

    did_anything = False

    if args.all or args.goog_recent:
        did_anything = True
        print("Downloading GOOG recent history ...", flush=True)
        path = download_goog_recent_history(output_root=datasets_root)
        print(f"Saved: {path}")
        time.sleep(0.05)

    if args.all or args.goog_long:
        did_anything = True
        print("Downloading GOOG long (max) history ...", flush=True)
        path = download_goog_long_history(output_root=datasets_root)
        print(f"Saved: {path}")
        time.sleep(0.05)

    if args.all or args.mts_repo:
        did_anything = True
        print("Downloading multivariate time-series datasets repo ...", flush=True)
        path = download_multivariate_time_series_repo(output_root=datasets_root)
        print(f"Extracted to: {path}")
        time.sleep(0.05)

    if not did_anything:
        print("No action selected. Use --all or one of: --goog-recent, --goog-long, --mts-repo", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


