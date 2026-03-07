"""
universe.py
Option contract universe definitions and live-API contract fetching.

This module defines:

* ``BucketSpec`` — an immutable dataclass describing a slice of the options
  universe by contract type, DTE range, and moneyness band.
* ``BEST_BUCKET``  — puts 30–60 DTE, moneyness 0.92–0.99 (OTM).
  Historically the CRR surface provides accurate pricing here.
* ``WORST_BUCKET`` — calls 120–150 DTE, near-ATM (moneyness 0.98–1.02).
  CRR pricing is less accurate here due to vol-smile / skew effects.
* ``OptionUniverseBuilder`` — live Polygon API contract fetcher that filters
  by a ``BucketSpec`` on a given as-of date and spot price.

For cached / offline operation see ``data_cache.CachedUniverseBuilder``, which
exposes the same ``list_contracts`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from polygon import RESTClient


# ─────────────────────────────────────────────────────────────────────────────
# Bucket specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BucketSpec:
    """
    Immutable specification for a slice of the options universe.

    Attributes
    ----------
    label : str
        Human-readable identifier, e.g. ``"PUT_30_60_OTM"``.
    option_type : str
        ``"put"`` or ``"call"``.
    min_dte : int
        Minimum days-to-expiry (inclusive) for contracts in this bucket.
    max_dte : int
        Maximum days-to-expiry (inclusive) for contracts in this bucket.
    min_moneyness : float
        Minimum moneyness ``K / S`` (inclusive).
    max_moneyness : float
        Maximum moneyness ``K / S`` (inclusive).
    """

    label: str
    option_type: str
    min_dte: int
    max_dte: int
    min_moneyness: float
    max_moneyness: float


# ─────────────────────────────────────────────────────────────────────────────
# Pre-defined buckets
# ─────────────────────────────────────────────────────────────────────────────

BEST_BUCKET = BucketSpec(
    label="PUT_30_60_OTM",
    option_type="put",
    min_dte=30,
    max_dte=60,
    min_moneyness=0.92,
    max_moneyness=0.99,
)
"""
Short-dated OTM put bucket.

The CRR surface historically provides a tight fit in this region — puts
with 30–60 DTE trade with high volume, narrow spreads, and a well-defined
smile that the four-parameter surface model captures accurately. This is the
*primary* research bucket.
"""

WORST_BUCKET = BucketSpec(
    label="CALL_120_150_ATM",
    option_type="call",
    min_dte=120,
    max_dte=150,
    min_moneyness=0.98,
    max_moneyness=1.02,
)
"""
Long-dated near-ATM call bucket.

CRR pricing is less reliable here: long-dated near-ATM calls are sensitive
to the volatility term structure and forward skew, which the simple polynomial
surface model does not fully capture. This bucket serves as the *control*
group in the research to contrast against ``BEST_BUCKET``.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Live contract fetcher
# ─────────────────────────────────────────────────────────────────────────────

class OptionUniverseBuilder:
    """
    Fetches the set of tradable option contracts from the Polygon API
    that satisfy a ``BucketSpec`` filter on a given date.

    Parameters
    ----------
    api_key : str
        Polygon.io API key.
    underlying_symbol : str
        Underlying equity ticker. Defaults to ``"SPY"``.
    """

    def __init__(self, api_key: str, underlying_symbol: str = "SPY") -> None:
        self.client = RESTClient(api_key=api_key)
        self.underlying_symbol = underlying_symbol

    def list_contracts(
        self,
        as_of: str,
        spot: float,
        bucket: BucketSpec,
    ) -> pd.DataFrame:
        """
        Return all option contracts that satisfy ``bucket`` on ``as_of``.

        Parameters
        ----------
        as_of : str
            Reference date in ``YYYY-MM-DD`` format. Used to compute DTE and
            to filter out contracts that had not yet listed.
        spot : float
            Underlying spot price on ``as_of``. Used to compute moneyness and
            apply the ``bucket.min_moneyness`` / ``bucket.max_moneyness``
            filter.
        bucket : BucketSpec
            Universe slice definition.

        Returns
        -------
        pd.DataFrame
            Columns: ``option_ticker``, ``strike``, ``expiry``
            (``pd.Timestamp``), ``contract_type``, ``bucket``,
            ``moneyness``. Returns an empty DataFrame if no contracts match.
        """
        exp_gte = (pd.Timestamp(as_of) + pd.Timedelta(days=bucket.min_dte)).strftime("%Y-%m-%d")
        exp_lte = (pd.Timestamp(as_of) + pd.Timedelta(days=bucket.max_dte)).strftime("%Y-%m-%d")

        rows = []
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

            if ticker is None or expiry is None or not np.isfinite(float(strike)):
                continue

            moneyness = float(strike) / float(spot)
            if bucket.min_moneyness <= moneyness <= bucket.max_moneyness:
                rows.append({
                    "option_ticker": ticker,
                    "strike": float(strike),
                    "expiry": pd.Timestamp(expiry),
                    "contract_type": bucket.option_type,
                    "bucket": bucket.label,
                    "moneyness": moneyness,
                })

        return pd.DataFrame(rows)
