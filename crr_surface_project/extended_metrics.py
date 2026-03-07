"""
extended_metrics.py
Comprehensive performance metrics for the institutional report.
All functions accept the equity and trades DataFrames returned by run_backtest().
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Risk-adjusted return metrics
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(daily_pnl: pd.Series, periods_per_year: int = 252) -> float:
    std = daily_pnl.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float((daily_pnl.mean() / std) * math.sqrt(periods_per_year))


def sortino_ratio(
    daily_pnl: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio using downside deviation below target."""
    excess = daily_pnl - target_return
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    downside_std = math.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return float("nan")
    return float((daily_pnl.mean() / downside_std) * math.sqrt(periods_per_year))


def calmar_ratio(daily_pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return / |max drawdown in dollars|."""
    cum = daily_pnl.cumsum()
    dd = (cum - cum.cummax()).min()
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    annualized_return = daily_pnl.mean() * periods_per_year
    return float(annualized_return / abs(dd))


def omega_ratio(daily_pnl: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega ratio: probability-weighted gains above threshold /
    probability-weighted losses below threshold.
    """
    gains = (daily_pnl[daily_pnl > threshold] - threshold).sum()
    losses = (threshold - daily_pnl[daily_pnl < threshold]).sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def information_ratio(strategy_pnl: pd.Series, benchmark_pnl: pd.Series) -> float:
    """
    (Strategy daily return - Benchmark daily return) / tracking error.
    Both series must have aligned DatetimeIndex.
    """
    tracking = strategy_pnl - benchmark_pnl.reindex(strategy_pnl.index, fill_value=0.0)
    std = tracking.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(tracking.mean() / std * math.sqrt(252))


# ─────────────────────────────────────────────────────────────────────────────
# Drawdown analysis
# ─────────────────────────────────────────────────────────────────────────────

def drawdown_series(cum_pnl: pd.Series) -> pd.Series:
    """Returns underwater curve in dollars (always <= 0)."""
    return cum_pnl - cum_pnl.cummax()


def drawdown_table(equity: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Identifies top N drawdown episodes.
    Returns DataFrame: start_date, trough_date, end_date, max_dd, duration_days, recovery_days
    """
    cum = equity.set_index("date")["cum_pnl"] if "date" in equity.columns else equity["cum_pnl"]
    dd = drawdown_series(cum)

    episodes = []
    in_dd = False
    start_date = None
    peak_val = None

    for date, val in dd.items():
        if not in_dd and val < 0:
            in_dd = True
            start_date = date
            peak_val = cum.loc[date] - val  # running max at this point
            trough_date = date
            trough_val = val
        elif in_dd:
            if val < trough_val:
                trough_date = date
                trough_val = val
            if val == 0:
                episodes.append({
                    "start_date": start_date,
                    "trough_date": trough_date,
                    "end_date": date,
                    "max_drawdown": float(trough_val),
                    "duration_days": (date - start_date).days,
                    "recovery_days": (date - trough_date).days,
                })
                in_dd = False

    # Handle open drawdown at end of series
    if in_dd:
        episodes.append({
            "start_date": start_date,
            "trough_date": trough_date,
            "end_date": None,
            "max_drawdown": float(trough_val),
            "duration_days": (dd.index[-1] - start_date).days,
            "recovery_days": None,
        })

    if not episodes:
        return pd.DataFrame(columns=["start_date", "trough_date", "end_date",
                                     "max_drawdown", "duration_days", "recovery_days"])

    df = pd.DataFrame(episodes)
    df = df.sort_values("max_drawdown").head(top_n).reset_index(drop=True)
    return df


def max_drawdown_duration(equity: pd.DataFrame) -> int:
    """Length in calendar days of the longest drawdown episode."""
    table = drawdown_table(equity, top_n=100)
    if table.empty:
        return 0
    return int(table["duration_days"].max())


# ─────────────────────────────────────────────────────────────────────────────
# Rolling metrics
# ─────────────────────────────────────────────────────────────────────────────

def rolling_sharpe(
    daily_pnl: pd.Series, window: int = 30, periods_per_year: int = 252
) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    roll_mean = daily_pnl.rolling(window).mean()
    roll_std = daily_pnl.rolling(window).std(ddof=1)
    return (roll_mean / roll_std * math.sqrt(periods_per_year)).rename("rolling_sharpe")


def rolling_volatility(
    daily_pnl: pd.Series, window: int = 30, periods_per_year: int = 252
) -> pd.Series:
    """Rolling annualized volatility of daily P&L."""
    return (daily_pnl.rolling(window).std(ddof=1) * math.sqrt(periods_per_year)).rename("rolling_vol")


def rolling_returns(daily_pnl: pd.Series, window: int = 21) -> pd.Series:
    """Rolling total return over a window."""
    return daily_pnl.rolling(window).sum().rename("rolling_return")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark comparison
# ─────────────────────────────────────────────────────────────────────────────

def spy_buy_hold_pnl(spy_prices: pd.DataFrame, initial_capital: float = 100_000.0) -> pd.Series:
    """
    Daily P&L for a SPY buy-and-hold position using initial_capital.
    spy_prices: DataFrame with DatetimeIndex and 'close' column.
    """
    closes = spy_prices["close"].sort_index()
    shares = initial_capital / float(closes.iloc[0])
    daily_ret = closes.pct_change().fillna(0.0)
    return (daily_ret * shares * closes.shift(1).fillna(closes.iloc[0])).rename("spy_bh_pnl")


def benchmark_metrics(
    strategy_equity: pd.DataFrame,
    spy_prices: pd.DataFrame,
    risk_free_rate: float = 0.045,
) -> Dict:
    """
    Computes cross-benchmark metrics: IR, beta, correlation, relative returns.
    """
    strat_pnl = strategy_equity.set_index("date")["daily_pnl"] if "date" in strategy_equity.columns \
        else strategy_equity["daily_pnl"]

    spy_closes = spy_prices["close"].sort_index()
    spy_daily_ret = spy_closes.pct_change().fillna(0.0)
    spy_daily_pnl = spy_buy_hold_pnl(spy_prices)

    aligned = pd.DataFrame({
        "strat": strat_pnl,
        "spy_pnl": spy_daily_pnl.reindex(strat_pnl.index, fill_value=0.0),
        "spy_ret": spy_daily_ret.reindex(strat_pnl.index, fill_value=0.0),
    }).dropna()

    if len(aligned) < 10:
        return {}

    strat = aligned["strat"]
    spy = aligned["spy_pnl"]

    corr = float(strat.corr(spy)) if spy.std() > 0 else float("nan")
    beta = float(strat.cov(spy) / spy.var()) if spy.var() > 0 else float("nan")

    strat_total = float(strat.sum())
    spy_total = float(spy.sum())

    return {
        "strategy_sharpe": sharpe_ratio(strat),
        "benchmark_sharpe": sharpe_ratio(spy),
        "information_ratio": information_ratio(strat, spy),
        "strategy_total_pnl": strat_total,
        "benchmark_total_pnl": spy_total,
        "correlation_with_spy": corr,
        "beta_to_spy": beta,
        "strategy_sortino": sortino_ratio(strat),
        "benchmark_sortino": sortino_ratio(spy),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Monthly / yearly breakdown
# ─────────────────────────────────────────────────────────────────────────────

def monthly_pnl_table(equity: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot table: rows = Year, columns = Jan..Dec + Annual.
    Values = total P&L for that month.
    """
    df = equity.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    df["year"] = df.index.year
    df["month"] = df.index.month

    monthly = df.groupby(["year", "month"])["daily_pnl"].sum().unstack("month")
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly = monthly.rename(columns=month_names)
    monthly["Annual"] = monthly.sum(axis=1)
    return monthly


def yearly_summary(equity: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Per-year summary: total_pnl, sharpe, sortino, calmar, max_dd, win_rate.
    """
    df = equity.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year

    rows = []
    for year, grp in df.groupby("year"):
        pnl = grp["daily_pnl"].fillna(0.0)
        cum = pnl.cumsum()
        dd = drawdown_series(cum)
        rows.append({
            "year": year,
            "total_pnl": float(pnl.sum()),
            "sharpe": sharpe_ratio(pnl, periods_per_year),
            "sortino": sortino_ratio(pnl, periods_per_year=periods_per_year),
            "calmar": calmar_ratio(pnl, periods_per_year),
            "max_drawdown": float(dd.min()),
            "win_rate": float((pnl > 0).mean()),
            "trading_days": len(pnl),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Trade-level analytics
# ─────────────────────────────────────────────────────────────────────────────

def trade_pnl_series(trades: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """
    Match ENTRY and EXIT rows in trades to compute per-trade P&L.
    Returns DataFrame: option_ticker, entry_date, exit_date, holding_days,
                       side, option_type, trade_pnl, entry_spot, exit_spot
    """
    if trades.empty:
        return pd.DataFrame()

    entries = trades[trades["action"].str.startswith("ENTRY")].copy()
    exits = trades[trades["action"] == "EXIT"].copy()

    records = []
    for _, entry in entries.iterrows():
        ticker = entry["option_ticker"]
        matching = exits[
            (exits["option_ticker"] == ticker) &
            (exits["date"] >= entry["date"])
        ].sort_values("date")

        if matching.empty:
            continue

        exit_row = matching.iloc[0]
        side = int(entry.get("side", 1))
        entry_price = float(entry.get("market_mid", entry.get("option_mid", np.nan)))
        exit_price = float(exit_row.get("option_mid", np.nan))

        if np.isfinite(entry_price) and np.isfinite(exit_price):
            trade_pnl = side * 1 * 100.0 * (exit_price - entry_price)
        else:
            trade_pnl = float("nan")

        holding = (exit_row["date"] - entry["date"]).days

        records.append({
            "option_ticker": ticker,
            "entry_date": entry["date"],
            "exit_date": exit_row["date"],
            "holding_days": holding,
            "side": side,
            "option_type": entry.get("option_type", ""),
            "trade_pnl": trade_pnl,
            "entry_spot": entry.get("spot", np.nan),
            "exit_spot": exit_row.get("spot", np.nan),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "bucket": entry.get("bucket", ""),
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def trade_statistics(trades: pd.DataFrame, equity: pd.DataFrame) -> Dict:
    """Comprehensive trade-level statistics."""
    if trades.empty:
        return {"total_trades": 0}

    per_trade = trade_pnl_series(trades, equity)
    entries = trades[trades["action"].str.startswith("ENTRY")]

    result: Dict = {
        "total_entries": int(len(entries)),
        "total_exits": int((trades["action"] == "EXIT").sum()),
        "long_entries": int((trades["action"] == "ENTRY_LONG_CHEAP").sum()),
        "short_entries": int((trades["action"] == "ENTRY_SHORT_RICH").sum()),
    }

    if per_trade.empty:
        return result

    pnl = per_trade["trade_pnl"].dropna()
    if len(pnl) == 0:
        return result

    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]

    result.update({
        "win_rate": float((pnl > 0).mean()),
        "avg_holding_days": float(per_trade["holding_days"].mean()),
        "profit_factor": float(winners.sum() / abs(losers.sum())) if len(losers) > 0 else float("inf"),
        "avg_winner_pnl": float(winners.mean()) if len(winners) > 0 else float("nan"),
        "avg_loser_pnl": float(losers.mean()) if len(losers) > 0 else float("nan"),
        "largest_win": float(pnl.max()),
        "largest_loss": float(pnl.min()),
        "median_trade_pnl": float(pnl.median()),
        "total_trade_pnl": float(pnl.sum()),
    })

    # By side
    for side_val, side_label in [(1, "long"), (-1, "short")]:
        side_pnl = per_trade.loc[per_trade["side"] == side_val, "trade_pnl"].dropna()
        if len(side_pnl) > 0:
            result[f"win_rate_{side_label}"] = float((side_pnl > 0).mean())
            result[f"avg_pnl_{side_label}"] = float(side_pnl.mean())

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance
# ─────────────────────────────────────────────────────────────────────────────

def alpha_ttest(daily_pnl: pd.Series) -> Dict:
    """
    Two-sided t-test: H0 = mean daily P&L is zero.
    Returns t_stat, p_value, is_significant, annualized_alpha_estimate.
    Uses scipy.stats.ttest_1samp if available, else manual computation.
    """
    clean = daily_pnl.dropna()
    n = len(clean)
    if n < 10:
        return {"t_stat": float("nan"), "p_value": float("nan"),
                "is_significant_5pct": False, "annualized_alpha": float("nan")}

    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(clean, 0)
    except ImportError:
        # Manual t-test
        mean = float(clean.mean())
        std = float(clean.std(ddof=1))
        t_stat = mean / (std / math.sqrt(n))
        # Approximate p-value using normal distribution for large n
        import math as _math
        p_value = 2.0 * (1.0 - 0.5 * _math.erfc(-abs(t_stat) / _math.sqrt(2)))

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "is_significant_5pct": bool(p_value < 0.05),
        "annualized_alpha": float(clean.mean() * 252),
        "n_observations": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Market regime analysis
# ─────────────────────────────────────────────────────────────────────────────

def vix_regime_breakdown(
    equity: pd.DataFrame,
    vix_series: Optional[pd.Series],
) -> pd.DataFrame:
    """
    Segments trading days by VIX regime: low (<15), medium (15-25), high (>25).
    vix_series: pd.Series indexed by date with VIX close values. If None, returns empty.
    """
    if vix_series is None or vix_series.empty:
        return pd.DataFrame()

    df = equity.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    df["vix"] = vix_series.reindex(df.index)
    df = df.dropna(subset=["vix"])

    bins = [0, 15, 25, float("inf")]
    labels = ["Low VIX (<15)", "Medium VIX (15-25)", "High VIX (>25)"]
    df["regime"] = pd.cut(df["vix"], bins=bins, labels=labels)

    rows = []
    for regime, grp in df.groupby("regime", observed=True):
        pnl = grp["daily_pnl"].fillna(0.0)
        rows.append({
            "regime": str(regime),
            "days": len(pnl),
            "total_pnl": float(pnl.sum()),
            "sharpe": sharpe_ratio(pnl),
            "win_rate": float((pnl > 0).mean()),
            "avg_daily_pnl": float(pnl.mean()),
        })

    return pd.DataFrame(rows)


def trend_regime_breakdown(equity: pd.DataFrame, spy_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Segments into trending (SPY > 20-day MA) vs. mean-reverting (SPY <= 20-day MA).
    """
    df = equity.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    closes = spy_prices["close"].sort_index()
    ma20 = closes.rolling(20).mean()
    df["above_ma"] = (closes.reindex(df.index) > ma20.reindex(df.index))

    rows = []
    for regime_name, regime_mask in [("Trending (SPY>MA20)", True), ("Mean-Reverting (SPY≤MA20)", False)]:
        grp = df[df["above_ma"] == regime_mask]
        pnl = grp["daily_pnl"].fillna(0.0)
        if len(pnl) == 0:
            continue
        rows.append({
            "regime": regime_name,
            "days": len(pnl),
            "total_pnl": float(pnl.sum()),
            "sharpe": sharpe_ratio(pnl),
            "win_rate": float((pnl > 0).mean()),
            "avg_daily_pnl": float(pnl.mean()),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Master summary
# ─────────────────────────────────────────────────────────────────────────────

def full_performance_summary(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    spy_prices: pd.DataFrame,
    risk_free_rate: float = 0.045,
    vix_series: Optional[pd.Series] = None,
) -> Dict:
    """
    Computes all performance metrics and returns a single nested dict.
    This is the primary input to the reporting module.
    """
    df = equity.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    pnl = df["daily_pnl"].fillna(0.0)
    cum = pnl.cumsum()
    dd = drawdown_series(cum)

    return {
        "overview": {
            "start_date": str(df.index.min().date()),
            "end_date": str(df.index.max().date()),
            "trading_days": len(pnl),
            "total_pnl": float(cum.iloc[-1]),
            "mean_daily_pnl": float(pnl.mean()),
        },
        "risk_adjusted": {
            "sharpe": sharpe_ratio(pnl),
            "sortino": sortino_ratio(pnl),
            "calmar": calmar_ratio(pnl),
            "omega": omega_ratio(pnl),
        },
        "drawdown": {
            "max_drawdown": float(dd.min()),
            "max_drawdown_duration_days": max_drawdown_duration(equity),
            "drawdown_table": drawdown_table(equity, top_n=5),
        },
        "rolling": {
            "sharpe_30d": rolling_sharpe(pnl, window=30),
            "vol_30d": rolling_volatility(pnl, window=30),
            "return_21d": rolling_returns(pnl, window=21),
        },
        "benchmark": benchmark_metrics(equity, spy_prices, risk_free_rate),
        "monthly": monthly_pnl_table(equity),
        "yearly": yearly_summary(equity),
        "trades": trade_statistics(trades, equity),
        "trade_detail": trade_pnl_series(trades, equity),
        "alpha_test": alpha_ttest(pnl),
        "vix_regime": vix_regime_breakdown(equity, vix_series),
        "trend_regime": trend_regime_breakdown(equity, spy_prices),
    }
