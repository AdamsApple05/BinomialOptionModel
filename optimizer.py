"""
optimizer.py
------------
Random Search (Monte Carlo Parameter) Optimizer for the Volatility Arbitrage Strategy.
Finds the optimal hyperparameters by running randomized configurations in parallel.
"""

import os
import random
import concurrent.futures
import pandas as pd
from typing import Dict, Any

from backtest_engine import OptionBacktester

# 1. The Control Asset (We optimize against our best performer)
CONTROL_CONFIG = {
    "label": "SPY 10% OTM Call 180D (Control)",
    "symbol": "SPY",
    "option_ticker": "O:SPY230331C00405000",
    "strike": 405.0,
    "expiry_days": 181,
    "option_type": "call",
    "start_date": "2022-10-01",
    "end_date": "2023-03-31"
}

# 2. Define the Parameter Space Bounds
PARAM_SPACE = {
    "z_entry_sell":       (1.0, 3.0),    # Explore tighter vs looser entry
    "z_entry_buy":        (2.0, 4.0),
    # Explore quick exits vs riding to the mean
    "z_exit":             (0.1, 1.0),
    # Explore fast (10d) vs slow (45d) baselines
    "lookback":           (10, 45),
    "put_skew_offset":    (0.2, 1.0),
    "max_trade_drawdown": (-0.10, -0.25)  # Explore tight stops vs wide stops
}

# Fixed parameters that we don't want to randomize
STATIC_PARAMS = {
    "initial_capital":    1_000_000.0,
    "risk_free_rate":     0.04,
    "max_hold_days":      21,
    "stock_tc_bps":       0.0,
    "binomial_steps":     50,
    "capital_allocation": 0.95,
}


def generate_random_params() -> Dict[str, Any]:
    """Generates a single randomized parameter set from the bounds."""
    return {
        "z_entry_sell":       round(random.uniform(*PARAM_SPACE["z_entry_sell"]), 2),
        "z_entry_buy":        round(random.uniform(*PARAM_SPACE["z_entry_buy"]), 2),
        "z_exit":             round(random.uniform(*PARAM_SPACE["z_exit"]), 2),
        "lookback":           random.randint(*PARAM_SPACE["lookback"]),
        "put_skew_offset":    round(random.uniform(*PARAM_SPACE["put_skew_offset"]), 2),
        "max_trade_drawdown": round(random.uniform(*PARAM_SPACE["max_trade_drawdown"]), 2),
    }


def run_optimization_iteration(iteration_id: int, params: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Runs a single backtest with the randomized parameters."""
    backtester = OptionBacktester(
        api_key=api_key, initial_capital=STATIC_PARAMS["initial_capital"])

    try:
        results = backtester.run_backtest(
            symbol=CONTROL_CONFIG["symbol"],
            strike=CONTROL_CONFIG["strike"],
            expiry_days=CONTROL_CONFIG["expiry_days"],
            start_date=CONTROL_CONFIG["start_date"],
            end_date=CONTROL_CONFIG["end_date"],
            option_type=CONTROL_CONFIG["option_type"],
            option_ticker=CONTROL_CONFIG["option_ticker"],
            risk_free_rate=STATIC_PARAMS["risk_free_rate"],
            z_entry_sell=params["z_entry_sell"],
            z_entry_buy=params["z_entry_buy"],
            z_exit=params["z_exit"],
            lookback=params["lookback"],
            max_hold_days=STATIC_PARAMS["max_hold_days"],
            max_trade_drawdown=params["max_trade_drawdown"],
            put_skew_offset=params["put_skew_offset"],
            stock_tc_bps=STATIC_PARAMS["stock_tc_bps"],
            binomial_steps=STATIC_PARAMS["binomial_steps"],
            capital_allocation=STATIC_PARAMS["capital_allocation"],
        )
        return {"id": iteration_id, "params": params, "success": True, "results": results}
    except Exception as e:
        return {"id": iteration_id, "params": params, "success": False, "error": str(e)}


def main():
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "POLYGON_API_KEY environment variable is not set.")

    NUM_ITERATIONS = 1000  # How many random variations to test
    print(
        f"Starting Parameter Optimization: {NUM_ITERATIONS} random configurations...")

    max_workers = min(NUM_ITERATIONS, os.cpu_count() or 4)
    successful_runs = []

    # Generate the random configurations
    configurations = [(i, generate_random_params())
                      for i in range(NUM_ITERATIONS)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_optimization_iteration, config[0], config[1], api_key): config
            for config in configurations
        }

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res["success"]:
                successful_runs.append(res)
                print(
                    f"Iteration {res['id']:03d} | Sharpe: {res['results'].get('portfolio_sharpe', 0):.2f} | Return: {res['results'].get('portfolio_return', 0):.2%}")
            else:
                print(f"Iteration {res['id']:03d} | FAILED: {res['error']}")

    # 3. Analyze and Rank the Results
    if not successful_runs:
        print("No successful runs to analyze.")
        return

    # Compile data into a DataFrame for easy sorting
    data = []
    for run in successful_runs:
        row = run["params"].copy()
        row.update({
            "Sharpe": run["results"].get("portfolio_sharpe", 0.0),
            "Return": run["results"].get("portfolio_return", 0.0),
            "MaxDD": run["results"].get("max_drawdown", 0.0),
            "Trades": run["results"].get("num_trades", 0)
        })
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by the objective function: Highest Sharpe Ratio
    df = df.sort_values(by="Sharpe", ascending=False)

    print("\n" + "="*80)
    print("🏆 TOP 5 PARAMETER CONFIGURATIONS 🏆")
    print("="*80)

    # Print the top 5 results nicely formatted
    top_5 = df.head(5)
    for idx, row in top_5.iterrows():
        print(
            f"\nRank {list(top_5.index).index(idx) + 1} (Sharpe: {row['Sharpe']:.2f}, Return: {row['Return']:.2%}, MaxDD: {row['MaxDD']:.2%})")
        print(f"  - Lookback:      {int(row['lookback'])} days")
        print(f"  - Z-Entry Sell:  {row['z_entry_sell']:.2f}")
        print(f"  - Z-Exit:        {row['z_exit']:.2f}")
        print(f"  - Stop Loss:     {row['max_trade_drawdown']:.0%}")
        print(f"  - Put Skew Adj:  {row['put_skew_offset']:.2f}")


if __name__ == "__main__":
    main()
