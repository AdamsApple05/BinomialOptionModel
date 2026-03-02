"""
main.py
-------
Entry point for the Delta-Neutral Z-Score Mispricing Backtest (Parallelized).
Executes multiple configuration targets against historical data.
"""

from __future__ import annotations
import os
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

from backtest_engine import OptionBacktester

# Option chain universe targeted for 90+ DTE vega-dominant contracts
CONFIGS = [
    {"label": "SPY 10% OTM Put 180D", "symbol": "SPY", "option_ticker": "O:SPY230331P00330000", "strike": 330.0,
        "expiry_days": 181, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-31"},
    {"label": "SPY 10% OTM Call 180D", "symbol": "SPY", "option_ticker": "O:SPY230331C00405000", "strike": 405.0,
        "expiry_days": 181, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-31"},
    {"label": "QQQ 15% OTM Put 167D", "symbol": "QQQ", "option_ticker": "O:QQQ230317P00230000", "strike": 230.0,
        "expiry_days": 167, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "IWM 10% OTM Put 180D", "symbol": "IWM", "option_ticker": "O:IWM230317P00155000", "strike": 155.0,
        "expiry_days": 167, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "DIA 10% OTM Call 180D", "symbol": "DIA", "option_ticker": "O:DIA230317C00320000", "strike": 320.0,
        "expiry_days": 167, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "NVDA 15% OTM Put 167D", "symbol": "NVDA", "option_ticker": "O:NVDA230317P00105000", "strike": 105.0,
        "expiry_days": 167, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "AAPL 10% OTM Call 167D", "symbol": "AAPL", "option_ticker": "O:AAPL230317C00155000", "strike": 155.0,
        "expiry_days": 167, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "MSFT 10% OTM Call 180D", "symbol": "MSFT", "option_ticker": "O:MSFT230317C00265000", "strike": 265.0,
        "expiry_days": 167, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "AMZN 15% OTM Put 180D", "symbol": "AMZN", "option_ticker": "O:AMZN230317P00095000", "strike": 95.0,
        "expiry_days": 167, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "TSLA 15% OTM Call 167D", "symbol": "TSLA", "option_ticker": "O:TSLA230317C00275000", "strike": 275.0,
        "expiry_days": 167, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "XLF 10% OTM Put 180D", "symbol": "XLF", "option_ticker": "O:XLF230317P00028000", "strike": 28.0,
        "expiry_days": 167, "option_type": "put", "start_date": "2022-10-01", "end_date": "2023-03-17"},
    {"label": "XLE 10% OTM Call 180D", "symbol": "XLE", "option_ticker": "O:XLE230317C00090000", "strike": 90.0,
        "expiry_days": 167, "option_type": "call", "start_date": "2022-10-01", "end_date": "2023-03-17"},
]

SHARED = {
    "initial_capital":    1_000_000.0,
    "risk_free_rate":     0.04,
    "z_entry_sell":       1.16,
    "z_entry_buy":        2.5,
    "z_exit":             0.75,
    "lookback":           23,
    "max_hold_days":      21,
    "max_trade_drawdown": -0.11,
    "put_skew_offset":    0.21,
    "stock_tc_bps":       0.0,
    "binomial_steps":     50,
    "capital_allocation": 0.2,
}


def run_single_backtest(cfg: Dict[str, Any], shared: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    backtester = OptionBacktester(
        api_key=api_key,
        initial_capital=shared["initial_capital"],
    )

    try:
        results = backtester.run_backtest(
            symbol=cfg["symbol"],
            strike=cfg["strike"],
            expiry_days=cfg["expiry_days"],
            start_date=cfg["start_date"],
            end_date=cfg["end_date"],
            option_type=cfg["option_type"],
            option_ticker=cfg["option_ticker"],
            risk_free_rate=shared["risk_free_rate"],
            z_entry_sell=shared["z_entry_sell"],
            z_entry_buy=shared["z_entry_buy"],
            z_exit=shared["z_exit"],
            lookback=shared["lookback"],
            max_hold_days=shared["max_hold_days"],
            max_trade_drawdown=shared["max_trade_drawdown"],
            put_skew_offset=shared["put_skew_offset"],
            stock_tc_bps=shared["stock_tc_bps"],
            binomial_steps=shared["binomial_steps"],
            capital_allocation=shared["capital_allocation"],
        )

        safe_label = "".join([c if c.isalnum() else "_" for c in cfg["label"]])
        backtester.export_to_csv(filename_prefix=safe_label)

        return {
            "label": cfg["label"],
            "success": True,
            "results": results,
            "equity_curve": backtester.equity_curve,
            "error": None
        }
    except Exception as exc:
        return {
            "label": cfg["label"],
            "success": False,
            "results": None,
            "equity_curve": None,
            "error": str(exc)
        }


def plot_from_data(equity_curve: list[Dict], initial_capital: float, title: str) -> None:
    if not equity_curve:
        return

    eq = pd.DataFrame(equity_curve)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(eq["date"], eq["equity"],
                 label="Portfolio Equity", linewidth=1.5)
    axes[0].axhline(initial_capital, linestyle="--",
                    color="grey", label="Initial Capital")
    axes[0].set_title(title)
    axes[0].set_ylabel("Equity ($)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    equity_s = eq.set_index("date")["equity"]
    dd = (equity_s - equity_s.cummax()) / equity_s.cummax() * 100
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.4,
                         color="red", label="Drawdown %")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "POLYGON_API_KEY environment variable is not set.\n"
            "Run:  export POLYGON_API_KEY='your_key_here'"
        )

    print(f"Starting parallel backtest with {len(CONFIGS)} configurations...")
    max_workers = min(len(CONFIGS), os.cpu_count() or 4)
    completed_runs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_backtest, cfg, SHARED, api_key): cfg
            for cfg in CONFIGS
        }

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            print(f"\n{'='*60}\n  Results: {res['label']}\n{'='*60}")

            if not res["success"]:
                print(f"  [FAILED] {res['error']}")
                continue

            completed_runs.append(res)

            labels = {
                "total_pnl":         "Total P&L ($)",
                "num_trades":        "Closed trades",
                "win_rate":          "Win rate",
                "avg_pnl":           "Avg P&L / trade ($)",
                "sharpe_ratio":      "Trade Sharpe",
                "max_win":           "Max win ($)",
                "max_loss":          "Max loss ($)",
                "portfolio_return":  "Portfolio return",
                "portfolio_sharpe":  "Annualised Sharpe",
                "max_drawdown":      "Max drawdown",
            }
            for key, label in labels.items():
                val = res["results"].get(key, "N/A")
                if isinstance(val, float):
                    if key in ("win_rate", "portfolio_return", "max_drawdown"):
                        print(f"  {label:<28} {val:>10.2%}")
                    else:
                        print(f"  {label:<28} {val:>10.2f}")
                else:
                    print(f"  {label:<28} {val!s:>10}")

    for run in completed_runs:
        plot_from_data(run["equity_curve"],
                       SHARED["initial_capital"], run["label"])


if __name__ == "__main__":
    main()
