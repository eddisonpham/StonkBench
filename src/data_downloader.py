"""
Utility script to download and save OHLC stock data using yfinance, for given tickers.
"""

import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf


def download_stock_history(ticker: str) -> Path | None:
    """Download full OHLC history for a single ticker, only OHLC, no extra rows."""
    output_dir = Path("data/raw") / ticker.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{ticker.upper()}.csv"

    if output_csv.exists():
        print(f"[SKIP] {ticker}: already exists at {output_csv}")
        return output_csv

    print(f"[DOWNLOADING] {ticker} ...", flush=True)
    df = yf.download(ticker, period="max", interval="1d", auto_adjust=False, progress=False)

    if df is None or df.empty:
        print(f"[WARNING] No data returned for {ticker}. Skipping.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].title() for col in df.columns]
    else:
        df.columns = [c.title() for c in df.columns]

    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    df_ohlc = df.loc[:, df.columns.intersection(ohlc_cols)].copy()

    if df_ohlc.empty:
        print(f"[WARNING] {ticker}: OHLC columns missing. Skipping.")
        return None

    df_ohlc.to_csv(output_csv, index=False, header=True)

    print(f"[SAVED] {ticker}: {len(df_ohlc)} rows → {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Download OHLC data via Yahoo Finance (yfinance).")
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="List of stock tickers to download (e.g. AAPL MSFT GOOG)."
    )
    args = parser.parse_args()

    for ticker in args.tickers:
        download_stock_history(ticker)


if __name__ == "__main__":
    main()
