#!/usr/bin/env bash
# Download daily close + volume via src/data_downloader.py (yfinance).
#
# Edit the arrays and dates below, or override at runtime:
#   START=2020-01-01 END=2024-01-01 ./download_data.sh
#   ./download_data.sh --start 2020-01-01 --end 2024-01-01 SPY AAPL
#   TICKERS="SPY QQQ" ./download_data.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- date range (yfinance end date is exclusive) ---
START="${START:-2023-01-01}"
END="${END:-2025-01-01}"

# --- ETFs ---
ETFS=(
  SPY   # S&P 500
  QQQ   # Nasdaq-100
  IWM   # Russell 2000
  XLF   # Financials
  XLV   # Healthcare
)

# --- stocks ---
STOCKS=(
  AAPL MSFT NVDA AVGO JPM LLY UNH AMZN TSLA
  CAT UNP META NFLX PG COST XOM CVX NEE PLD LIN
)

usage() {
  cat <<'EOF'
Usage: ./download_data.sh [options] [TICKER ...]

Options:
  --start YYYY-MM-DD   Start date (default: START env or script default)
  --end YYYY-MM-DD     End date, exclusive (default: END env or script default)
  -h, --help           Show this help

If TICKER(s) are given on the command line, only those are downloaded.
Otherwise uses ETFS + STOCKS from this script, or TICKERS env if set.

Environment:
  START, END, TICKERS (space-separated), PYTHON (default: python)
EOF
}

PYTHON="${PYTHON:-python}"
TICKERS_ENV="${TICKERS:-}"
CLI_TICKERS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end)   END="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; CLI_TICKERS+=("$@"); break ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    *) CLI_TICKERS+=("$1"); shift ;;
  esac
done

if [[ ${#CLI_TICKERS[@]} -gt 0 ]]; then
  SELECTED=("${CLI_TICKERS[@]}")
elif [[ -n "$TICKERS_ENV" ]]; then
  # shellcheck disable=SC2206
  SELECTED=($TICKERS_ENV)
else
  SELECTED=("${ETFS[@]}" "${STOCKS[@]}")
fi

echo "Downloading ${#SELECTED[@]} tickers: ${SELECTED[*]}"
echo "Date range: ${START} -> ${END} (end exclusive)"

exec "$PYTHON" src/data_downloader.py \
  --start "$START" \
  --end "$END" \
  --index "${SELECTED[@]}"
