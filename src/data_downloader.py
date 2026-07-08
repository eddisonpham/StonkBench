"""Download daily close and volume from Yahoo Finance for stocks and ETFs.

Run from repo root:

    python src/data_downloader.py --start 2015-01-01 --end 2025-01-01 --index SPY QQQ IWM XLF XLV AAPL MSFT NVDA AVGO JPM LLY UNH AMZN TSLA CAT UNP META NFLX PG COST XOM CVX NEE PLD LIN
"""

from pathlib import Path
import argparse
from typing import List, Optional

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path("/data") if Path("/data").exists() else PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "combined_data.csv"


def download_ticker(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download daily close and volume for one ticker."""
    ticker = ticker.upper()
    hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        print(f"[ERROR] No data for {ticker}")
        return None

    dates = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
    return pd.DataFrame(
        {
            "timestamp": dates.strftime("%Y%m%d").astype(int),
            ticker: hist["Close"].to_numpy(),
            f"{ticker}_volume": hist["Volume"].to_numpy(),
        }
    )


def years_to_range(years: List[str]) -> tuple[str, str]:
    years_int = sorted(int(y) for y in years)
    return f"{years_int[0]}-01-01", f"{years_int[-1] + 1}-01-01"


def merge_on_timestamp(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="timestamp", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download daily close and volume for stocks/ETFs via yfinance."
    )
    parser.add_argument(
        "--index",
        "-i",
        "--ticker",
        "-t",
        nargs="+",
        dest="tickers",
        default=["SPY"],
        help="Ticker symbols (stocks or ETFs)",
    )
    parser.add_argument("--year", "-y", nargs="+", default=["2023", "2024"])
    parser.add_argument("--start", help="Start date YYYY-MM-DD (overrides --year)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (exclusive, like yfinance)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.start:
        start = args.start
        end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    else:
        start, end = years_to_range(args.year)

    tickers = [t.upper() for t in args.tickers]
    dfs = [df for t in tickers if (df := download_ticker(t, start, end)) is not None]

    if not dfs:
        print("[ERROR] No data downloaded.")
        return

    merged = merge_on_timestamp(dfs)
    if merged.empty:
        print("[ERROR] No overlapping dates across tickers.")
        return

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(merged)} rows x {len(merged.columns)} cols to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
