"""
walk_forward_engine.py
---------------------
Orchestrates rolling optimization and out-of-sample trading.
Automatically retrieves POLYGON_API_KEY from the environment.
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta
from backtest_engine import OptionBacktester
from optimizer import generate_random_params


class WalkForwardEngine:
    def __init__(self, symbol: str, base_config: dict):
        """
        Initializes the engine using the API key stored in environment variables.
        """
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "POLYGON_API_KEY not found in environment variables.")

        self.symbol = symbol
        self.base_config = base_config
        self.results = []

    def run_wfa(self, start_date: str, end_date: str, train_months: int = 6, test_months: int = 1):
        """
        Executes the Walk-Forward Analysis loop.
        """
        current_start = pd.to_datetime(start_date)
        final_limit = pd.to_datetime(end_date)

        while current_start + timedelta(days=(train_months + test_months) * 30) <= final_limit:
            train_end = current_start + timedelta(days=train_months * 30)
            test_end = train_end + timedelta(days=test_months * 30)

            # 1. IN-SAMPLE OPTIMIZATION
            print(
                f"\n[WFA] Optimizing: {current_start.date()} to {train_end.date()}")
            best_params = self._optimize(current_start.strftime(
                '%Y-%m-%d'), train_end.strftime('%Y-%m-%d'))

            # 2. OUT-OF-SAMPLE TRADING
            if best_params:
                print(
                    f"[WFA] Trading OOS: {train_end.date()} to {test_end.date()}")
                print(
                    f"      Params: Lookback {best_params['lookback']}d, Z-Sell {best_params['z_entry_sell']}")

                oos_metrics = self._trade(train_end.strftime(
                    '%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), best_params)

                self.results.append({
                    "test_period": f"{train_end.date()} to {test_end.date()}",
                    "params": best_params,
                    "metrics": oos_metrics
                })

            # Roll forward by the test period length
            current_start = current_start + timedelta(days=test_months * 30)

    def _optimize(self, start: str, end: str):
        """Finds the best parameters with an internal buffer."""
        # Pad the start date by 45 days to ensure Z-Score calculation is possible
        buffer_start = (pd.to_datetime(start) -
                        pd.Timedelta(days=45)).strftime('%Y-%m-%d')

        iterations = [generate_random_params() for _ in range(30)]
        best_sharpe = -np.inf
        best_p = None

        tester = OptionBacktester(api_key=self.api_key)
        for p in iterations:
            try:
                res = tester.run_backtest(
                    symbol=self.symbol,
                    strike=self.base_config['strike'],
                    expiry_days=self.base_config['expiry_days'],
                    start_date=buffer_start,  # Buffer used here
                    end_date=end,
                    option_type=self.base_config['option_type'],
                    option_ticker=self.base_config['option_ticker'],
                    **p
                )
                if res['portfolio_sharpe'] > best_sharpe:
                    best_sharpe = res['portfolio_sharpe']
                    best_p = p
            except:
                continue
        return best_p

    def _trade(self, start: str, end: str, params: dict):
        """
        Calculates a buffer to ensure the 'aligned trading days' 
        check passes in the backtest engine.
        """
        start_dt = pd.to_datetime(start)

        # We need a buffer larger than the lookback to ensure the check passes.
        # 60 days covers the max possible lookback (45) + safety margin.
        buffer_days = max(60, params.get('lookback', 15) + 15)
        buffer_start = (start_dt - pd.Timedelta(days=buffer_days)
                        ).strftime('%Y-%m-%d')

        tester = OptionBacktester(api_key=self.api_key)
        return tester.run_backtest(
            symbol=self.symbol,
            strike=self.base_config['strike'],
            expiry_days=self.base_config['expiry_days'],
            start_date=buffer_start,
            end_date=end,
            option_type=self.base_config['option_type'],
            option_ticker=self.base_config['option_ticker'],
            **params
        )


# Execution block
if __name__ == "__main__":
    # Ensure your terminal has the key set: export POLYGON_API_KEY='your_key'
    config = {
        "strike": 405.0,
        "expiry_days": 181,
        "option_type": "call",
        "option_ticker": "O:SPY230331C00405000"
    }

    wfa = WalkForwardEngine("SPY", config)
    # Note: Ensure date range covers at least (train_months + test_months)
    wfa.run_wfa("2022-01-01", "2023-06-01")
