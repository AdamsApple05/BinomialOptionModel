"""
market_data.py
--------------
Thin wrapper around the Polygon.io REST API (polygon-api-client).

Install:
    pip install polygon-api-client

The public `polygon` package exposes `RESTClient` at the top level:
    from polygon import RESTClient
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from polygon import RESTClient  # pip install polygon-api-client


@dataclass
class MarketData:
    """
    Fetch underlying OHLCV and option bid/ask data from Polygon.io.

    Parameters
    ----------
    api_key : str
        Your Polygon.io API key.
    """

    api_key: str
    _client: RESTClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = RESTClient(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Underlying
    # ------------------------------------------------------------------

    def get_underlying_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Return daily OHLCV for *symbol* between *start_date* and *end_date*.

        Returns
        -------
        DataFrame indexed by date with columns:
            open, high, low, close, volume
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
            rows.append(
                {
                    "date": pd.to_datetime(agg.timestamp, unit="ms").tz_localize(None),
                    "open": agg.open,
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": agg.volume,
                }
            )

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        df.index = pd.to_datetime(df.index)
        return df

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    def option_daily_close_quote(
        self,
        option_ticker: str,
        start_date: str,
        end_date: str,
        tz: str = "America/New_York",
    ) -> pd.DataFrame:
        """
        Fetches daily closing aggregates for the option contract.

        Returns
        -------
        DataFrame indexed by date with columns:
            bid, ask, mid, spread, bid_size, ask_size
        """
        if not option_ticker.startswith("O:"):
            option_ticker = "O:" + option_ticker

        rows = []
        # Use list_aggs instead of list_quotes to get clean daily OHLC
        for agg in self._client.list_aggs(
            ticker=option_ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=50_000,
        ):
            rows.append(
                {
                    "date": pd.to_datetime(agg.timestamp, unit="ms").tz_localize(None),
                    "close": agg.close,
                }
            )

        if not rows:
            return pd.DataFrame(columns=["bid", "ask", "mid", "spread", "bid_size", "ask_size"])

        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        df.index = pd.to_datetime(df.index)

        # Approximate the bid/ask and mid using the daily closing price
        # (Assuming a 1% spread for backtesting simulation)
        df["mid"] = df["close"]
        df["bid"] = df["close"] * 0.99
        df["ask"] = df["close"] * 1.01
        df["spread"] = df["ask"] - df["bid"]
        df["bid_size"] = 10
        df["ask_size"] = 10

        # Return ONLY the required option columns so 'close' does not collide with the underlying's 'close'
        return df[["bid", "ask", "mid", "spread", "bid_size", "ask_size"]]
