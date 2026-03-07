"""
run_download.py
Populates the local data cache from the Polygon API.

Usage (from project root):
    python crr_surface_project/run_download.py

Or from within the crr_surface_project/ directory:
    python run_download.py

Set POLYGON_API_KEY environment variable (or edit config.py) before running.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from config import GlobalConfig
from universe import BEST_BUCKET, WORST_BUCKET
from data_cache import DataCache, RateLimiter, PolygonDownloader

# ── Configuration ─────────────────────────────────────────────────────────────

START_YEAR = 2022
END_YEAR = 2024
SYMBOL = "SPY"

# Place the data directory next to this script file
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"


def get_trading_dates(underlying_frames: dict[int, pd.DataFrame]) -> list[pd.Timestamp]:
    """Return sorted list of all trading dates from underlying data."""
    frames = [df for df in underlying_frames.values() if not df.empty]
    if not frames:
        return []
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return list(combined.index)


def get_spot_series(underlying_frames: dict[int, pd.DataFrame]) -> pd.Series:
    """Return a Series indexed by date with close prices."""
    frames = [df for df in underlying_frames.values() if not df.empty]
    if not frames:
        return pd.Series(dtype=float)
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined["close"]


def collect_unique_tickers(cache: DataCache, buckets, trading_dates: list[pd.Timestamp]) -> set[str]:
    """Read all cached contract list CSVs and collect unique option tickers."""
    tickers: set[str] = set()
    for bucket in buckets:
        for date in trading_dates:
            date_str = date.strftime("%Y-%m-%d")
            if cache.has_contract_list(bucket.label, date_str):
                df = cache.read_contract_list(bucket.label, date_str)
                if not df.empty and "option_ticker" in df.columns:
                    tickers.update(df["option_ticker"].tolist())
    return tickers


def main():
    cfg = GlobalConfig()

    if cfg.api_key == "YOUR_API_KEY_HERE":
        raise RuntimeError("Please set POLYGON_API_KEY environment variable.")

    print(f"Data directory: {DATA_DIR}")
    print(f"Downloading {SYMBOL} data for {START_YEAR}-{END_YEAR}")
    print(f"Rate limit: {cfg.calls_per_minute} calls/min")
    print()

    cache = DataCache(DATA_DIR)
    rl = RateLimiter(calls_per_minute=cfg.calls_per_minute)
    downloader = PolygonDownloader(api_key=cfg.api_key, cache=cache, rate_limiter=rl)

    buckets = [BEST_BUCKET, WORST_BUCKET]

    # ── Step 1: Download underlying OHLC ──────────────────────────────────────
    print("=== Step 1: Underlying OHLC ===")
    underlying_frames: dict[int, pd.DataFrame] = {}
    for year in range(START_YEAR, END_YEAR + 1):
        df = downloader.download_underlying_year(SYMBOL, year)
        underlying_frames[year] = df

    all_trading_dates = get_trading_dates(underlying_frames)
    spot_series = get_spot_series(underlying_frames)
    print(f"Total trading dates: {len(all_trading_dates)}")
    print()

    # ── Step 2: Download contract lists ───────────────────────────────────────
    print("=== Step 2: Contract Lists ===")
    for bucket in buckets:
        print(f"Bucket: {bucket.label}")
        downloader.download_contract_lists(
            symbol=SYMBOL,
            bucket=bucket,
            trading_dates=all_trading_dates,
            spot_series=spot_series,
        )
    print()

    # ── Step 3: Collect unique option tickers ─────────────────────────────────
    print("=== Step 3: Collecting unique option tickers ===")
    unique_tickers = collect_unique_tickers(cache, buckets, all_trading_dates)
    print(f"Unique option tickers across all dates/buckets: {len(unique_tickers)}")
    print()

    # ── Step 4: Download option OHLC ──────────────────────────────────────────
    print("=== Step 4: Option OHLC ===")
    tickers_list = sorted(unique_tickers)
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Year {year}:")
        downloader.download_option_ohlc_batch(tickers=tickers_list, year=year)
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_underlying = sum(
        len(underlying_frames[y]) for y in range(START_YEAR, END_YEAR + 1)
        if y in underlying_frames and not underlying_frames[y].empty
    )
    total_contract_files = sum(
        1 for b in buckets
        for d in all_trading_dates
        if cache.has_contract_list(b.label, d.strftime("%Y-%m-%d"))
    )

    print("=== Download Complete ===")
    print(f"  Underlying bars:   {total_underlying}")
    print(f"  Contract list files: {total_contract_files}")
    print(f"  Unique option tickers: {len(unique_tickers)}")
    print(f"  Data stored in: {DATA_DIR}")


if __name__ == "__main__":
    main()
