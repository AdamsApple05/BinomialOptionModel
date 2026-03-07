import numpy as np
import pandas as pd


def safe_ts(x) -> pd.Timestamp:
    return pd.to_datetime(x).tz_localize(None) if getattr(pd.to_datetime(x), "tzinfo", None) else pd.to_datetime(x)


def annualized_hist_vol(close: pd.Series, trading_days_per_year: int = 252, lookback: int = 30) -> pd.Series:
    logret = np.log(close / close.shift(1))
    return logret.rolling(lookback).std() * np.sqrt(trading_days_per_year)


def compute_steps(T: float, min_steps: int = 25, max_steps: int = 300, steps_per_year: int = 252) -> int:
    raw = int(round(T * steps_per_year))
    return max(min_steps, min(max_steps, raw))
