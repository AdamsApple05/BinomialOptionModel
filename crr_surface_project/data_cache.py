"""
data_cache.py
Local CSV cache for Polygon API data + cached drop-ins for MarketData
and OptionUniverseBuilder.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from polygon import RESTClient

from market_data import MarketData
from universe import BucketSpec, OptionUniverseBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_ticker(ticker: str) -> str:
    """Make an option ticker safe for use as a filename.
    O:SPY240119P00400000  →  O_SPY240119P00400000
    """
    return ticker.replace(":", "_")


# ─────────────────────────────────────────────────────────────────────────────
# DataCache — manages directory layout and CSV I/O
# ─────────────────────────────────────────────────────────────────────────────

class DataCache:
    """
    Directory layout under cache_dir/:
        underlying/     {SYMBOL}_{YEAR}.csv
        contracts/      {BUCKET_LABEL}_{DATE}.csv
        options/{YEAR}/ {sanitized_ticker}.csv
    """

    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)
        self._setup_dirs()

    def _setup_dirs(self):
        (self.root / "underlying").mkdir(parents=True, exist_ok=True)
        (self.root / "contracts").mkdir(parents=True, exist_ok=True)
        (self.root / "options").mkdir(parents=True, exist_ok=True)

    # ── Underlying OHLC ──────────────────────────────────────────────────────

    def _und_path(self, symbol: str, year: int) -> Path:
        return self.root / "underlying" / f"{symbol}_{year}.csv"

    def has_underlying(self, symbol: str, year: int) -> bool:
        return self._und_path(symbol, year).exists()

    def read_underlying(self, symbol: str, year: int) -> pd.DataFrame:
        df = pd.read_csv(self._und_path(symbol, year), parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date")
        return df

    def write_underlying(self, df: pd.DataFrame, symbol: str, year: int):
        out = df.reset_index() if df.index.name == "date" else df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out.to_csv(self._und_path(symbol, year), index=False)

    # ── Contract lists ────────────────────────────────────────────────────────

    def _contract_path(self, bucket_label: str, date_str: str) -> Path:
        return self.root / "contracts" / f"{bucket_label}_{date_str}.csv"

    def has_contract_list(self, bucket_label: str, date_str: str) -> bool:
        return self._contract_path(bucket_label, date_str).exists()

    def read_contract_list(self, bucket_label: str, date_str: str) -> pd.DataFrame:
        _empty = pd.DataFrame(columns=["option_ticker", "strike", "expiry",
                                       "contract_type", "bucket", "moneyness"])
        path = self._contract_path(bucket_label, date_str)
        try:
            df = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return _empty
        if df.empty or "expiry" not in df.columns:
            return _empty
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
        return df

    def write_contract_list(self, df: pd.DataFrame, bucket_label: str, date_str: str):
        df.to_csv(self._contract_path(bucket_label, date_str), index=False)

    # ── Option OHLC ──────────────────────────────────────────────────────────

    def _opt_path(self, ticker: str, year: int) -> Path:
        year_dir = self.root / "options" / str(year)
        year_dir.mkdir(exist_ok=True)
        return year_dir / f"{_sanitize_ticker(ticker)}.csv"

    def has_option_ohlc(self, ticker: str, year: int) -> bool:
        return self._opt_path(ticker, year).exists()

    def read_option_ohlc(self, ticker: str, year: int) -> pd.DataFrame:
        df = pd.read_csv(self._opt_path(ticker, year), parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date")
        return df

    def write_option_ohlc(self, df: pd.DataFrame, ticker: str, year: int):
        out = df.reset_index() if df.index.name == "date" else df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out.to_csv(self._opt_path(ticker, year), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# RateLimiter — token-bucket throttle
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Enforces a minimum interval between API calls."""

    def __init__(self, calls_per_minute: int = 5):
        self.min_interval = 60.0 / max(calls_per_minute, 1)
        self._last: float = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# PolygonDownloader — populates the cache from the API
# ─────────────────────────────────────────────────────────────────────────────

class PolygonDownloader:
    """Downloads Polygon data with caching, rate limiting, and retry logic."""

    def __init__(self, api_key: str, cache: DataCache, rate_limiter: RateLimiter):
        self._api_key = api_key
        self._client = RESTClient(api_key=api_key)
        self.cache = cache
        self.rl = rate_limiter

    def download_underlying_year(self, symbol: str, year: int) -> pd.DataFrame:
        """Download one calendar year of daily OHLC. Skips if already cached."""
        if self.cache.has_underlying(symbol, year):
            print(f"  [cache] {symbol} {year}")
            return self.cache.read_underlying(symbol, year)

        start, end = f"{year}-01-01", f"{year}-12-31"
        self.rl.wait()
        rows = []
        for agg in self._client.list_aggs(
            ticker=symbol, multiplier=1, timespan="day",
            from_=start, to=end, limit=50_000,
        ):
            rows.append({
                "date": pd.to_datetime(agg.timestamp, unit="ms").tz_localize(None).normalize(),
                "open": agg.open, "high": agg.high,
                "low": agg.low, "close": agg.close,
                "volume": getattr(agg, "volume", None),
            })

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        df.index = pd.to_datetime(df.index)
        self.cache.write_underlying(df, symbol, year)
        print(f"  [downloaded] {symbol} {year}: {len(df)} bars")
        return df

    def download_contract_lists(
        self,
        symbol: str,
        bucket: BucketSpec,
        trading_dates: List[pd.Timestamp],
        spot_series: pd.Series,
    ) -> None:
        """
        For each trading date, fetch options contracts matching the bucket spec
        and cache the result. Skips dates that are already cached.
        spot_series: pd.Series indexed by Timestamp → close price.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kw: x

        skipped = 0
        downloaded = 0
        for date in tqdm(trading_dates, desc=f"Contracts {bucket.label}", leave=True):
            date_str = date.strftime("%Y-%m-%d")
            if self.cache.has_contract_list(bucket.label, date_str):
                skipped += 1
                continue

            if date not in spot_series.index:
                continue
            spot = float(spot_series.loc[date])

            exp_gte = (date + pd.Timedelta(days=bucket.min_dte)
                       ).strftime("%Y-%m-%d")
            exp_lte = (date + pd.Timedelta(days=bucket.max_dte)
                       ).strftime("%Y-%m-%d")

            self.rl.wait()
            rows = []
            try:
                for c in self._client.list_options_contracts(
                    underlying_ticker=symbol,
                    contract_type=bucket.option_type,
                    as_of=date_str,
                    expiration_date_gte=exp_gte,
                    expiration_date_lte=exp_lte,
                    limit=1000,
                ):
                    ticker = getattr(c, "ticker", None)
                    strike = getattr(c, "strike_price", np.nan)
                    expiry = getattr(c, "expiration_date", None)
                    if ticker is None or expiry is None:
                        continue
                    try:
                        strike_f = float(strike)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(strike_f):
                        continue
                    m = strike_f / spot
                    if bucket.min_moneyness <= m <= bucket.max_moneyness:
                        rows.append({
                            "option_ticker": ticker,
                            "strike": strike_f,
                            "expiry": str(pd.Timestamp(expiry).normalize()),
                            "contract_type": bucket.option_type,
                            "bucket": bucket.label,
                            "moneyness": m,
                        })
            except Exception as e:
                print(f"\n  Warning: failed {date_str}: {e}")
                continue

            df = pd.DataFrame(rows)
            self.cache.write_contract_list(df, bucket.label, date_str)
            downloaded += 1

        print(f"  {bucket.label}: {downloaded} downloaded, {skipped} already cached")

    def download_option_ohlc_batch(
        self,
        tickers: List[str],
        year: int,
        retry_max: int = 3,
        max_workers: int = 50,
    ) -> None:
        """
        Download full-year OHLC for each option ticker in parallel.

        Uses a ``ThreadPoolExecutor`` to issue many concurrent HTTP requests,
        eliminating the network-latency bottleneck of sequential downloads.
        Each worker thread owns its own ``RESTClient`` instance to avoid
        shared-state issues. File writes are safe because every ticker maps
        to a unique file path.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kw: x

        start, end = f"{year}-01-01", f"{year}-12-31"
        api_key = self._api_key
        cache = self.cache

        # Separate work into needed vs. already cached up front
        pending = []
        skipped = 0
        for ticker in tickers:
            ticker_key = ticker if ticker.startswith("O:") else "O:" + ticker
            if cache.has_option_ohlc(ticker_key, year):
                skipped += 1
            else:
                pending.append(ticker_key)

        downloaded = 0
        failed = 0

        def _fetch_one(ticker_key: str) -> tuple[str, bool]:
            """Worker: download one ticker and write to cache. Returns (ticker, success)."""
            client = RESTClient(api_key=api_key)  # thread-local client
            for attempt in range(retry_max):
                try:
                    rows = []
                    for agg in client.list_aggs(
                        ticker=ticker_key, multiplier=1, timespan="day",
                        from_=start, to=end, limit=50_000,
                    ):
                        rows.append({
                            "date": pd.to_datetime(agg.timestamp, unit="ms").tz_localize(None).normalize(),
                            "open": agg.open, "high": agg.high,
                            "low": agg.low, "close": agg.close,
                            "volume": getattr(agg, "volume", None),
                        })

                    if rows:
                        df = pd.DataFrame(rows).sort_values(
                            "date").set_index("date")
                        df.index = pd.to_datetime(df.index)
                    else:
                        df = pd.DataFrame(
                            columns=["open", "high", "low", "close", "volume"])
                        df.index.name = "date"

                    cache.write_option_ohlc(df, ticker_key, year)
                    return ticker_key, True

                except Exception as exc:
                    if attempt < retry_max - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return ticker_key, False

        if pending:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_fetch_one, t): t for t in pending}
                for future in tqdm(
                    as_completed(futures),
                    total=len(pending),
                    desc=f"Options OHLC {year}",
                    leave=True,
                ):
                    _, success = future.result()
                    if success:
                        downloaded += 1
                    else:
                        failed += 1

        print(
            f"  Year {year}: {downloaded} downloaded, {skipped} cached, {failed} failed")


# ─────────────────────────────────────────────────────────────────────────────
# CachedMarketData — drop-in replacement for MarketData
# ─────────────────────────────────────────────────────────────────────────────

class CachedMarketData:
    """
    Drop-in replacement for MarketData with identical public interface.
    Reads from DataCache; falls back to live API on cache miss.
    """

    def __init__(self, api_key: str, cache: DataCache):
        self._live = MarketData(api_key=api_key)
        self.cache = cache

    def get_underlying_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        years = list(range(start_ts.year, end_ts.year + 1))

        frames = []
        for year in years:
            if self.cache.has_underlying(symbol, year):
                frames.append(self.cache.read_underlying(symbol, year))
            else:
                df = self._live.get_underlying_daily(
                    symbol, f"{year}-01-01", f"{year}-12-31")
                if not df.empty:
                    self.cache.write_underlying(df, symbol, year)
                frames.append(df)

        non_empty = [f for f in frames if not f.empty]
        if not non_empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        combined = pd.concat(non_empty)
        combined = combined[~combined.index.duplicated(
            keep="first")].sort_index()
        mask = (combined.index >= start_ts) & (combined.index <= end_ts)
        return combined.loc[mask]

    def get_option_daily_aggs(
        self,
        option_ticker: str,
        start_date: str,
        end_date: str,
        assumed_spread_bps: float = 25.0,
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        years = list(range(start_ts.year, end_ts.year + 1))

        ticker_key = option_ticker if option_ticker.startswith(
            "O:") else "O:" + option_ticker

        raw_frames = []
        for year in years:
            if self.cache.has_option_ohlc(ticker_key, year):
                raw_frames.append(
                    self.cache.read_option_ohlc(ticker_key, year))
            else:
                # Fall back to live for this year
                df_live = self._live.get_option_daily_aggs(
                    ticker_key, f"{year}-01-01", f"{year}-12-31",
                    assumed_spread_bps=assumed_spread_bps,
                )
                if not df_live.empty:
                    # Store raw OHLC (mid = close)
                    raw = pd.DataFrame({
                        "open": df_live.get("open", df_live["mid"]),
                        "high": df_live.get("high", df_live["mid"]),
                        "low": df_live.get("low", df_live["mid"]),
                        "close": df_live["mid"],
                        "volume": df_live["volume"],
                    }, index=df_live.index)
                    self.cache.write_option_ohlc(raw, ticker_key, year)
                    raw_frames.append(raw)

        if not raw_frames:
            return pd.DataFrame(columns=["bid", "ask", "mid", "spread", "close", "volume"])

        non_empty = [f for f in raw_frames if not f.empty]
        if not non_empty:
            return pd.DataFrame(columns=["bid", "ask", "mid", "spread", "close", "volume"])

        raw = pd.concat(non_empty)
        raw = raw[~raw.index.duplicated(keep="first")].sort_index()
        mask = (raw.index >= start_ts) & (raw.index <= end_ts)
        raw_slice = raw.loc[mask]

        if raw_slice.empty:
            return pd.DataFrame(columns=["bid", "ask", "mid", "spread", "close", "volume"])

        # Synthesize bid/ask
        spread_frac = float(assumed_spread_bps) / 10_000.0
        out = pd.DataFrame(index=raw_slice.index)
        out["mid"] = raw_slice["close"].astype(float)
        out["bid"] = out["mid"] * (1.0 - spread_frac / 2.0)
        out["ask"] = out["mid"] * (1.0 + spread_frac / 2.0)
        out["spread"] = out["ask"] - out["bid"]
        out["close"] = raw_slice["close"].astype(float)
        out["volume"] = raw_slice["volume"]
        return out[["bid", "ask", "mid", "spread", "close", "volume"]]


# ─────────────────────────────────────────────────────────────────────────────
# CachedUniverseBuilder — drop-in replacement for OptionUniverseBuilder
# ─────────────────────────────────────────────────────────────────────────────

class CachedUniverseBuilder:
    """
    Drop-in replacement for OptionUniverseBuilder with identical public interface.
    Reads contract lists from DataCache; falls back to live API on cache miss.
    """

    def __init__(self, api_key: str, cache: DataCache, underlying_symbol: str = "SPY"):
        self._live = OptionUniverseBuilder(
            api_key=api_key, underlying_symbol=underlying_symbol)
        self.cache = cache

    def list_contracts(self, as_of: str, spot: float, bucket: BucketSpec) -> pd.DataFrame:
        if self.cache.has_contract_list(bucket.label, as_of):
            df = self.cache.read_contract_list(bucket.label, as_of)
            if df.empty or "expiry" not in df.columns:
                return df
            df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
            return df

        # Fall back to live
        df = self._live.list_contracts(as_of=as_of, spot=spot, bucket=bucket)
        if not df.empty:
            self.cache.write_contract_list(df, bucket.label, as_of)
        return df
