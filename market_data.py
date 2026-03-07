from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from polygon import RESTClient


@dataclass
class MarketData:
    api_key: str
    _client: RESTClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = RESTClient(api_key=self.api_key)

    def get_underlying_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
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
