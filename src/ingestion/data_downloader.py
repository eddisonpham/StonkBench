"""
Dataset Downloader

download_goog_history: GOOG stock price history (CSV)

The outputs are saved under their respective subfolders.
"""

from pathlib import Path
import yfinance as yf

def download_goog_history(period: str = "5y",
                          interval: str = "1d") -> Path:
    """Download history for a Google (GOOG) ticker to CSV.

    - Uses yfinance for robust data access.
    - Default period is 5 years of daily candles.

    Returns the CSV file path.
    """
    ticker = "GOOG"
    output_dir = Path("data/raw") / f"{ticker}"
    output_dir.mkdir(exist_ok=True)
    output_csv = output_dir / f"{ticker}.csv"

    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period=period, interval=interval, auto_adjust=False)

    if df is None or df.empty:
        raise RuntimeError("No data returned from Yahoo Finance for the requested period.")

    df.to_csv(output_csv, index=True)
    return output_csv

if __name__ == "__main__":
    print("Downloading GOOG history ...", flush=True)
    path = download_goog_history()
    print(f"Saved: {path}")


