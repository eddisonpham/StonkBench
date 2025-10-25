"""
Utility script to download and save OHLC stock data using yfinance, for given tickers.
"""

from pathlib import Path
import argparse
import pandas as pd
import yfinance as yf


def download_stock_history(ticker: str, interval: str = "1d", period: str = "max") -> Path | None:
    """
    Download OHLC history for a single ticker.
    interval: '1d', '1m', '5m', etc.
    period: 'max', '7d', '60d', etc. (for 1m interval, max 7d)
    """
    output_dir = Path("data/raw") / ticker.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{ticker.upper()}_{interval}.csv"

    if output_csv.exists():
        print(f"[SKIP] {ticker}: already exists at {output_csv}")
        return output_csv

    print(f"[DOWNLOADING] {ticker} (interval={interval}, period={period}) ...", flush=True)
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return None

    if df is None or df.empty:
        print(f"[WARNING] No data returned for {ticker}. Skipping.")
        return None

    # Flatten MultiIndex columns if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].title() for col in df.columns]
    else:
        df.columns = [c.title() for c in df.columns]

    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    df_ohlc = df.loc[:, df.columns.intersection(ohlc_cols)].copy()

    if df_ohlc.empty:
        print(f"[WARNING] {ticker}: OHLC columns missing. Skipping.")
        return None

    df_ohlc.to_csv(output_csv, index=True, header=True)
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
    parser.add_argument(
        "--interval",
        default="1d",
        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"],
        help="Data interval (default 1d)"
    )
    parser.add_argument(
        "--period",
        default="max",
        help="Data period: e.g. 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"
    )
    args = parser.parse_args()

    # For 1-minute interval, restrict period to <=7d
    if args.interval == "1m" and args.period not in ["1d", "5d", "7d"]:
        print("[INFO] 1-minute interval only supports period <= 7d. Using period='7d'.")
        args.period = "7d"

    for ticker in args.tickers:
        download_stock_history(ticker, interval=args.interval, period=args.period)


if __name__ == "__main__":
    main()
