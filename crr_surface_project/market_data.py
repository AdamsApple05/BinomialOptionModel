"""
market_data.py
Polygon.io REST API client for fetching underlying and option daily OHLC data.

This module provides ``MarketData``, the live-API data provider used by the
strategy during backtests that do not use the local data cache. For
cached/offline operation see ``data_cache.CachedMarketData``, which exposes
an identical public interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from polygon import RESTClient


@dataclass
class MarketData:
    """
    Thin wrapper around the Polygon.io REST client.

    Provides two methods — ``get_underlying_daily`` and
    ``get_option_daily_aggs`` — that return consistently shaped DataFrames
    used throughout the strategy and backtest modules.

    Parameters
    ----------
    api_key : str
        Polygon.io API key.
    """

    api_key: str
    _client: RESTClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = RESTClient(api_key=self.api_key)

    # ── Public interface ──────────────────────────────────────────────────────

    def get_underlying_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for an underlying equity.

        Parameters
        ----------
        symbol : str
            Equity ticker, e.g. ``"SPY"``.
        start_date : str
            Inclusive start date in ``YYYY-MM-DD`` format.
        end_date : str
            Inclusive end date in ``YYYY-MM-DD`` format.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (timezone-naive), columns:
            ``open``, ``high``, ``low``, ``close``, ``volume``.
            Returns an empty DataFrame with those columns if no data is found.
        """
        rows = []
        for agg in self._client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=50_000,
        ):
            rows.append({
                "date": pd.to_datetime(agg.timestamp, unit="ms").tz_localize(None),
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": getattr(agg, "volume", None),
            })

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        df.index = pd.to_datetime(df.index)
        return df

    def get_option_daily_aggs(
        self,
        option_ticker: str,
        start_date: str,
        end_date: str,
        assumed_spread_bps: float = 25.0,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for an option contract and synthesise bid/ask.

        Because Polygon daily aggregates do not include separate bid and ask
        prices, a symmetric half-spread of ``assumed_spread_bps / 2`` basis
        points is applied around the closing price to produce ``bid`` and
        ``ask``.

        Parameters
        ----------
        option_ticker : str
            Option contract ticker with or without the ``"O:"`` prefix,
            e.g. ``"O:SPY240119P00400000"`` or ``"SPY240119P00400000"``.
        start_date : str
            Inclusive start date in ``YYYY-MM-DD`` format.
        end_date : str
            Inclusive end date in ``YYYY-MM-DD`` format.
        assumed_spread_bps : float
            Synthetic total bid-ask spread in basis points. Applied
            symmetrically: ``bid = mid * (1 - spread/2)``,
            ``ask = mid * (1 + spread/2)``.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (timezone-naive), columns:
            ``bid``, ``ask``, ``mid``, ``spread``, ``close``, ``volume``.
            Returns an empty DataFrame with those columns if no data is found.
        """
        if not option_ticker.startswith("O:"):
            option_ticker = "O:" + option_ticker

        rows = []
        for agg in self._client.list_aggs(
            ticker=option_ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=50_000,
        ):
            rows.append({
                "date": pd.to_datetime(agg.timestamp, unit="ms").tz_localize(None),
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": getattr(agg, "volume", None),
            })

        if not rows:
            return pd.DataFrame(columns=["bid", "ask", "mid", "spread", "close", "volume"])

        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        df.index = pd.to_datetime(df.index)

        df["mid"] = df["close"].astype(float)
        spread_frac = float(assumed_spread_bps) / 10_000.0
        df["bid"] = df["mid"] * (1.0 - spread_frac / 2.0)
        df["ask"] = df["mid"] * (1.0 + spread_frac / 2.0)
        df["spread"] = df["ask"] - df["bid"]

        return df[["bid", "ask", "mid", "spread", "close", "volume"]]
