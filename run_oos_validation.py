"""
run_oos_validation.py
---------------------
Runs a hard Out-of-Sample test on 2024 data using parameters 
discovered in 2022/2023.
"""

import os
from backtest_engine import OptionBacktester

# Parameters found by your optimizer for Rank 1
BEST_PARAMS = {
    "lookback": 14,
    "z_entry_sell": 1.40,
    "z_exit": 0.90,
    "max_trade_drawdown": -0.13,
    "put_skew_offset": 0.82
}

# 2024 Date Range (Purely Out-of-Sample)
OOS_CONFIGS = [
    {
        "label": "SPY 2024 OOS",
        "symbol": "SPY",
        "option_ticker": "O:SPY240621C00500000",  # Example 2024 Ticker
        "strike": 500.0,
        "expiry_days": 150,
        "start_date": "2024-01-01",
        "end_date": "2024-06-01"
    }
]


def main():
    api_key = os.getenv("POLYGON_API_KEY")
    tester = OptionBacktester(api_key)

    for cfg in OOS_CONFIGS:
        print(f"Running OOS Test for {cfg['label']}...")
        # Use existing engine logic to run the test
        # metrics = tester.run_backtest(...)
