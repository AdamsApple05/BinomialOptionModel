"""
main.py
-------
Consolidated Portfolio Version: Merges all backtest results into a single CSV.
"""

from __future__ import annotations
import os
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

from backtest_engine import OptionBacktester

# Universe configuration
CONFIGS = [
    {"label": "SPY_Put", "symbol": "SPY", "option_ticker": "O:SPY230331P00330000", "strike": 330.0,
        "expiry_days": 181, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-31"},
    {"label": "SPY_Call", "symbol": "SPY", "option_ticker": "O:SPY230331C00405000", "strike": 405.0,
        "expiry_days": 181, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-31"},
    {"label": "QQQ_Put", "symbol": "QQQ", "option_ticker": "O:QQQ230317P00230000", "strike": 230.0,
        "expiry_days": 167, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "IWM_Put", "symbol": "IWM", "option_ticker": "O:IWM230317P00155000", "strike": 155.0,
        "expiry_days": 167, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "DIA_Call", "symbol": "DIA", "option_ticker": "O:DIA230317C00320000", "strike": 320.0,
        "expiry_days": 167, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "NVDA_Put", "symbol": "NVDA", "option_ticker": "O:NVDA230317P00105000", "strike": 105.0,
        "expiry_days": 167, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "AAPL_Call", "symbol": "AAPL", "option_ticker": "O:AAPL230317C00155000", "strike": 155.0,
        "expiry_days": 167, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "MSFT_Call", "symbol": "MSFT", "option_ticker": "O:MSFT230317C00265000", "strike": 265.0,
        "expiry_days": 167, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "AMZN_Put", "symbol": "AMZN", "option_ticker": "O:AMZN230317P00095000", "strike": 95.0,
        "expiry_days": 167, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "TSLA_Call", "symbol": "TSLA", "option_ticker": "O:TSLA230317C00275000", "strike": 275.0,
        "expiry_days": 167, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "XLF_Put", "symbol": "XLF", "option_ticker": "O:XLF230317P00028000", "strike": 28.0,
        "expiry_days": 167, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "XLE_Call", "symbol": "XLE", "option_ticker": "O:XLE230317C00090000", "strike": 90.0,
        "expiry_days": 167, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-17"},
]

SHARED = {
    "initial_capital": 1_000_000.0,
    "risk_free_rate": 0.04,
    "z_entry_sell": 1.4,
    "z_entry_buy": 2.5,
    "z_exit": 0.7,
    "lookback": 18,
    "max_hold_days": 21,
    "max_trade_drawdown": -0.15,
    "put_skew_offset": 0.3,
    "stock_tc_bps": 0.0,
    "binomial_steps": 50,
    "capital_allocation": 0.2,
}


def run_single_backtest(cfg: Dict[str, Any], shared: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    backtester = OptionBacktester(
        api_key=api_key, initial_capital=shared["initial_capital"])
    try:
        results = backtester.run_backtest(
            symbol=cfg["symbol"], strike=cfg["strike"], expiry_days=cfg["expiry_days"],
            start_date=cfg["start_date"], end_date=cfg["end_date"], option_type=cfg["option_type"],
            option_ticker=cfg["option_ticker"], risk_free_rate=shared["risk_free_rate"],
            z_entry_sell=shared["z_entry_sell"], z_entry_buy=shared["z_entry_buy"],
            z_exit=shared["z_exit"], lookback=shared["lookback"],
            max_hold_days=shared["max_hold_days"], max_trade_drawdown=shared["max_trade_drawdown"],
            put_skew_offset=shared["put_skew_offset"], stock_tc_bps=shared["stock_tc_bps"],
            binomial_steps=shared["binomial_steps"], capital_allocation=shared["capital_allocation"]
        )
        return {"label": cfg["label"], "success": True, "results": results, "equity_curve": backtester.equity_curve}
    except Exception as exc:
        return {"label": cfg["label"], "success": False, "error": str(exc)}


def main() -> None:
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "POLYGON_API_KEY environment variable is not set.")

    print(f"Starting parallel backtest with {len(CONFIGS)} configurations...")
    completed_runs = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(
            run_single_backtest, cfg, SHARED, api_key): cfg for cfg in CONFIGS}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res["success"]:
                completed_runs.append(res)
                print(f"Completed: {res['label']}")
            else:
                print(f"Failed: {res['label']} - {res['error']}")

    # Consolidated CSV Export
    if completed_runs:
        all_curves = []
        for run in completed_runs:
            df = pd.DataFrame(run["equity_curve"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["equity"]].rename(
                columns={"equity": run["label"]})
            all_curves.append(df)

        # Merge all columns on date and save
        # The new, modern way to forward-fill missing data in Pandas
        # Explicitly set sort=True to align all dates correctly across the 12 configurations
        master_df = pd.concat(all_curves, axis=1,
                              sort=True).sort_index().ffill()
        os.makedirs("backtest_results", exist_ok=True)
        master_df.to_csv("backtest_results/master_portfolio_equity.csv")
        print(
            "\nMaster equity file saved to 'backtest_results/master_portfolio_equity.csv'")


if __name__ == "__main__":
    main()
