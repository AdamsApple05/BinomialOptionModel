from __future__ import annotations
import numpy as np
import pandas as pd


def compute_drawdown(cum_pnl: pd.Series) -> pd.Series:
    running_max = cum_pnl.cummax()
    return cum_pnl - running_max


def summarize_equity(equity: pd.DataFrame) -> dict:
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
    sharpe_like = np.nan if std == 0 or np.isnan(std) else (daily.mean() / std) * np.sqrt(252)

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
    rows = []
    for label, result in named_results.items():
        eq = summarize_equity(result["equity"])
        tr = summarize_trades(result["trades"])
        sg = summarize_signals(result["signals"])
        row = {"bucket": label}
        row.update(eq)
        row.update(tr)
        row.update(sg)
        rows.append(row)
    return pd.DataFrame(rows)
