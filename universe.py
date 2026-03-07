from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from polygon import RESTClient


@dataclass(frozen=True)
class BucketSpec:
    label: str
    option_type: str
    min_dte: int
    max_dte: int
    min_moneyness: float
    max_moneyness: float


BEST_BUCKET = BucketSpec(
    label="PUT_30_60_OTM",
    option_type="put",
    min_dte=30,
    max_dte=60,
    min_moneyness=0.92,
    max_moneyness=0.99,
)

WORST_BUCKET = BucketSpec(
    label="CALL_120_150_ATM",
    option_type="call",
    min_dte=120,
    max_dte=150,
    min_moneyness=0.98,
    max_moneyness=1.02,
)


class OptionUniverseBuilder:
    def __init__(self, api_key: str, underlying_symbol: str = "SPY"):
        self.client = RESTClient(api_key=api_key)
        self.underlying_symbol = underlying_symbol

    def list_contracts(self, as_of: str, spot: float, bucket: BucketSpec) -> pd.DataFrame:
        rows = []
        exp_gte = (pd.Timestamp(as_of) + pd.Timedelta(days=bucket.min_dte)).strftime("%Y-%m-%d")
        exp_lte = (pd.Timestamp(as_of) + pd.Timedelta(days=bucket.max_dte)).strftime("%Y-%m-%d")

        for c in self.client.list_options_contracts(
            underlying_ticker=self.underlying_symbol,
            contract_type=bucket.option_type,
            as_of=as_of,
            expiration_date_gte=exp_gte,
            expiration_date_lte=exp_lte,
            limit=1000,
        ):
            ticker = getattr(c, "ticker", None)
            strike = getattr(c, "strike_price", np.nan)
            expiry = getattr(c, "expiration_date", None)

            if ticker is None or expiry is None or not np.isfinite(strike):
                continue

            m = float(strike) / float(spot)
            if bucket.min_moneyness <= m <= bucket.max_moneyness:
                rows.append({
                    "option_ticker": ticker,
                    "strike": float(strike),
                    "expiry": pd.Timestamp(expiry),
                    "contract_type": bucket.option_type,
                    "bucket": bucket.label,
                    "moneyness": m,
                })

        return pd.DataFrame(rows)
