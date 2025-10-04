"""
Dataset Downloader

1) download_goog_history: GOOG stock price history (CSV)
3) download_multivariate_time_series_repo: Time-series datasets repo (zip -> extracted)

Usage examples (PowerShell / bash):
  python download.py --all
  python download.py --goog
  python download.py --mts-repo

The outputs are saved under their respective subfolders.
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
    path.mkdir(exist_ok=True)

def download_goog_history(ticker: str = "GOOG",
                          period: str = "5y",
                          interval: str = "1d") -> Path:
    """Download history for a Google (GOOG) ticker to CSV.

    - Uses yfinance for robust data access.
    - Default period is 5 years of daily candles.

    Returns the CSV file path.
    """
    output_dir = Path(f"{ticker}")
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

def download_multivariate_time_series_repo() -> Path:
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

    extraction_dir = Path("mvt-ts-data")
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

        gz_file_path.unlink()
        print(f"Removed {gz_file_path}")
    else:
        raise FileNotFoundError(f"{gz_file_path} does not exist...")

    return extraction_dir


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets into their respective directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--goog", action="store_true", help="Download GOOG history")
    parser.add_argument("--mts-repo", action="store_true", help="Download multivariate time-series repo")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    did_anything = False

    if args.all or args.goog:
        did_anything = True
        print("Downloading GOOG history ...", flush=True)
        path = download_goog_history()
        print(f"Saved: {path}")
        time.sleep(0.05)

    if args.all or args.mts_repo:
        did_anything = True
        print("Downloading multivariate time-series datasets repo ...", flush=True)
        path = download_multivariate_time_series_repo()
        print(f"Extracted to: {path}")
        time.sleep(0.05)

    if not did_anything:
        print("No action selected. Use --all or one of: --goog, --mts-repo", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


