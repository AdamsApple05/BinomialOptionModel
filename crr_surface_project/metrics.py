"""
metrics.py
Core performance metrics for the CRR Surface Delta-Hedge Strategy.

This module provides lightweight summary statistics computed directly from
the DataFrames returned by ``CRRSurfaceDeltaHedgeStrategy.run_backtest()``.
For the full institutional metrics suite (Sortino, Calmar, rolling Sharpe,
benchmark comparison, drawdown tables, regime analysis, etc.) see
``extended_metrics.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_drawdown(cum_pnl: pd.Series) -> pd.Series:
    """
    Compute the underwater (drawdown) curve in dollars.

    Parameters
    ----------
    cum_pnl : pd.Series
        Cumulative P&L series, e.g. ``equity["cum_pnl"]``.

    Returns
    -------
    pd.Series
        Drawdown series, always ``<= 0``. Each value is the distance in
        dollars below the running maximum of ``cum_pnl``.
    """
    running_max = cum_pnl.cummax()
    return cum_pnl - running_max


def summarize_equity(equity: pd.DataFrame) -> dict:
    """
    Compute headline equity-curve statistics.

    Parameters
    ----------
    equity : pd.DataFrame
        Output from ``run_backtest()["equity"]``. Must contain columns
        ``daily_pnl`` and ``cum_pnl``.

    Returns
    -------
    dict
        Keys: ``days``, ``total_pnl``, ``mean_daily_pnl``, ``std_daily_pnl``,
        ``sharpe_like`` (annualised Sharpe ratio using daily P&L units),
        ``max_drawdown`` (most negative value of the underwater curve),
        ``positive_day_rate`` (fraction of days with positive P&L).
        All numeric values are ``np.nan`` when the equity DataFrame is empty.
    """
    if equity.empty:
        return {
            "days": 0,
            "total_pnl": np.nan,
            "mean_daily_pnl": np.nan,
            "std_daily_pnl": np.nan,
            "sharpe_like": np.nan,
            "max_drawdown": np.nan,
            "positive_day_rate": np.nan,
        }

    daily = equity["daily_pnl"].fillna(0.0)
    cum_pnl = equity["cum_pnl"].fillna(0.0)
    dd = compute_drawdown(cum_pnl)

    std = daily.std(ddof=1)
    sharpe_like = (
        np.nan
        if std == 0 or np.isnan(std)
        else float((daily.mean() / std) * np.sqrt(252))
    )

    return {
        "days": int(len(equity)),
        "total_pnl": float(cum_pnl.iloc[-1]),
        "mean_daily_pnl": float(daily.mean()),
        "std_daily_pnl": float(std) if np.isfinite(std) else np.nan,
        "sharpe_like": float(sharpe_like) if np.isfinite(sharpe_like) else np.nan,
        "max_drawdown": float(dd.min()) if not dd.empty else np.nan,
        "positive_day_rate": float((daily > 0).mean()),
    }


def summarize_trades(trades: pd.DataFrame) -> dict:
    """
    Compute trade-count statistics from the transaction log.

    Parameters
    ----------
    trades : pd.DataFrame
        Output from ``run_backtest()["trades"]``. Must contain column
        ``action`` with values ``"ENTRY_LONG_CHEAP"``, ``"ENTRY_SHORT_RICH"``,
        and ``"EXIT"``.

    Returns
    -------
    dict
        Keys: ``entries``, ``exits``, ``long_entries``, ``short_entries``.
    """
    if trades.empty:
        return {
            "entries": 0,
            "exits": 0,
            "long_entries": 0,
            "short_entries": 0,
        }

    return {
        "entries": int(trades["action"].astype(str).str.startswith("ENTRY").sum()),
        "exits": int((trades["action"] == "EXIT").sum()),
        "long_entries": int((trades["action"] == "ENTRY_LONG_CHEAP").sum()),
        "short_entries": int((trades["action"] == "ENTRY_SHORT_RICH").sum()),
    }


def summarize_signals(signals: pd.DataFrame) -> dict:
    """
    Compute signal-observation statistics from the daily signal database.

    Parameters
    ----------
    signals : pd.DataFrame
        Output from ``run_backtest()["signals"]``. Must contain columns
        ``abs_residual_price``, ``abs_residual_iv``, ``surface_iv``, ``iv``.

    Returns
    -------
    dict
        Keys: ``signal_obs``, ``mean_abs_residual_price``,
        ``mean_abs_residual_iv``, ``mean_surface_iv``, ``mean_market_iv``.
    """
    if signals.empty:
        return {
            "signal_obs": 0,
            "mean_abs_residual_price": np.nan,
            "mean_abs_residual_iv": np.nan,
            "mean_surface_iv": np.nan,
            "mean_market_iv": np.nan,
        }

    return {
        "signal_obs": int(len(signals)),
        "mean_abs_residual_price": float(signals["abs_residual_price"].mean()),
        "mean_abs_residual_iv": float(signals["abs_residual_iv"].mean()),
        "mean_surface_iv": float(signals["surface_iv"].mean()),
        "mean_market_iv": float(signals["iv"].mean()),
    }


def compare_results(named_results: dict[str, dict]) -> pd.DataFrame:
    """
    Aggregate equity, trade, and signal summaries for multiple strategy runs.

    Parameters
    ----------
    named_results : dict[str, dict]
        Mapping of ``{label: run_backtest() output dict}`` for each bucket or
        parameter configuration being compared.

    Returns
    -------
    pd.DataFrame
        One row per entry in ``named_results``, with columns from
        ``summarize_equity``, ``summarize_trades``, and ``summarize_signals``,
        plus a leading ``bucket`` column.
    """
    rows = []
    for label, result in named_results.items():
        row = {"bucket": label}
        row.update(summarize_equity(result["equity"]))
        row.update(summarize_trades(result["trades"]))
        row.update(summarize_signals(result["signals"]))
        rows.append(row)
    return pd.DataFrame(rows)
