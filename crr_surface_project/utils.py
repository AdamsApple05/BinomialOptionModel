"""
utils.py
Shared utility functions used across the CRR Surface research framework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_ts(x) -> pd.Timestamp:
    """
    Convert any date-like object to a timezone-naive ``pd.Timestamp``.

    Parameters
    ----------
    x : date-like
        Any value accepted by ``pd.to_datetime``, including strings,
        ``datetime.date``, ``datetime.datetime``, or an existing
        ``pd.Timestamp``.

    Returns
    -------
    pd.Timestamp
        Timezone-naive timestamp. If the input is already timezone-aware,
        the timezone information is stripped via ``tz_localize(None)``.
    """
    ts = pd.to_datetime(x)
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_localize(None)
    return ts


def annualized_hist_vol(
    close: pd.Series,
    trading_days_per_year: int = 252,
    lookback: int = 30,
) -> pd.Series:
    """
    Compute rolling historical volatility from a price series.

    Volatility is estimated as the standard deviation of log-returns over a
    rolling window, scaled to annualised units by multiplying by
    ``sqrt(trading_days_per_year)``.

    Parameters
    ----------
    close : pd.Series
        Time-series of closing prices.
    trading_days_per_year : int
        Number of trading days used for annualisation. Defaults to ``252``.
    lookback : int
        Rolling window length in trading days. Defaults to ``30``.

    Returns
    -------
    pd.Series
        Annualised historical volatility. The first ``lookback - 1`` values
        are ``NaN`` due to the rolling window warm-up period.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(lookback).std() * np.sqrt(trading_days_per_year)


def compute_steps(
    T: float,
    min_steps: int = 25,
    max_steps: int = 300,
    steps_per_year: int = 252,
) -> int:
    """
    Derive a suitable CRR step count proportional to time-to-expiry.

    Maps the time-to-expiry ``T`` (in years) to a number of binomial tree
    steps, proportional to ``T * steps_per_year``, clamped to
    ``[min_steps, max_steps]``.

    Parameters
    ----------
    T : float
        Time-to-expiry in years.
    min_steps : int
        Minimum allowable step count. Defaults to ``25``.
    max_steps : int
        Maximum allowable step count. Defaults to ``300``.
    steps_per_year : int
        Base step density (steps per calendar year). Defaults to ``252``.

    Returns
    -------
    int
        Clamped step count suitable for use with ``BinomialOptionPricer``.
    """
    raw = int(round(T * steps_per_year))
    return max(min_steps, min(max_steps, raw))
