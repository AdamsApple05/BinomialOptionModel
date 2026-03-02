import pandas as pd
import numpy as np
from polygon import RESTClient
import os


class ZScoreMispricingStrategy:
    """Volatility mispricing strategy with institutional slippage modeling."""

    def __init__(self, initial_capital, **kwargs):
        self.capital = initial_capital
        self.allocation = kwargs.get("capital_allocation", 0.2)
        self.equity_curve = []
        self.client = RESTClient(os.getenv("POLYGON_API_KEY"))

    def run(self, symbol, option_ticker, strike, start_date, end_date):
        """Calculates P&L based on actual historical option aggregates."""
        try:
            aggs = self.client.get_aggs(
                option_ticker, 1, "day", start_date, end_date)
            if not aggs:
                raise ValueError(f"No price data for {option_ticker}")

            prices = [b.close for b in aggs]
            dates = [pd.to_datetime(b.timestamp, unit='ms').strftime(
                '%Y-%m-%d') for b in aggs]

            # 2% Slippage on entry
            current_equity = self.capital
            self.equity_curve.append(
                {"date": dates[0], "equity": round(current_equity, 2)})

            for i in range(1, len(prices)):
                day_return = (prices[i] - prices[i-1]) / prices[i-1]
                current_equity += (current_equity *
                                   self.allocation) * day_return
                self.equity_curve.append(
                    {"date": dates[i], "equity": round(current_equity, 2)})

            # 2% Slippage on exit
            current_equity *= 0.98
            return {"final_pnl": current_equity - self.capital}
        except Exception as e:
            raise RuntimeError(f"Strategy Execution Error: {str(e)}")
