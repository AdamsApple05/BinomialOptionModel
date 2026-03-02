"""
backtest_engine.py
------------------
Execution environment for the Z-Score Arbitrage strategy.
Fetches underlying/option data, iterates through market history, and compiles performance metrics.
"""

from __future__ import annotations
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from binomial_pricer import BinomialOptionPricer
from market_data import MarketData
from zscore_strategy import ZScoreMispricingStrategy


class OptionBacktester:
    def __init__(self, api_key: str, initial_capital: float = 1_000_000.0) -> None:
        self.data = MarketData(api_key=api_key)
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)

        self.equity_curve: list[Dict] = []
        self.trade_log: list[Dict] = []

    @staticmethod
    def _hist_vol(close: pd.Series, window: int = 30) -> pd.Series:
        """Calculates Exponentially Weighted Moving Average (EWMA) volatility."""
        log_ret = np.log(close / close.shift(1))
        return log_ret.ewm(span=window, min_periods=window).std() * np.sqrt(252)

    def _calc_current_equity(self, strategy: ZScoreMispricingStrategy, mid: float, S: float) -> float:
        """Calculates live Mark-to-Market (MtM) equity."""
        mtm = 0.0
        if strategy.position is not None:
            contracts = abs(strategy.position["contracts"])
            opt_val = mid * 100.0 * contracts
            stk_val = float(strategy.position["hedge_shares"]) * S

            if strategy.position["type"] == "short_option":
                mtm = float(
                    strategy.position["initial_proceeds"]) - opt_val + stk_val
            else:
                mtm = opt_val - \
                    float(strategy.position["initial_cost"]) + stk_val

        realised_pnl = float(np.sum(strategy.pnl_history))
        return self.capital + mtm + realised_pnl

    def run_backtest(
        self,
        symbol: str,
        strike: float,
        expiry_days: int,
        start_date: str,
        end_date: str,
        option_type: str = "call",
        option_ticker: str = "",
        risk_free_rate: float = 0.04,
        z_entry_sell: float = 1.5,
        z_entry_buy: float = 2.5,
        z_exit: float = 0.5,
        lookback: int = 15,
        max_hold_days: int = 21,
        max_trade_drawdown: float = -0.15,
        put_skew_offset: float = 0.5,
        stock_tc_bps: float = 0.0,
        binomial_steps: int = 50,
        capital_allocation: float = 0.95,
    ) -> Dict:

        if not option_ticker:
            raise ValueError("option_ticker is required.")

        # Require Vega-dominant options; avoid Gamma risk
        if expiry_days < 90:
            raise ValueError(
                f"DTE ({expiry_days}) < 90. Strategy requires Vega-dominant options.")

        # Establish EWMA volatility baseline
        vol_start_date = (pd.to_datetime(start_date) -
                          pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        stock = self.data.get_underlying_daily(
            symbol, vol_start_date, end_date)

        if stock.empty:
            raise ValueError(f"No underlying data returned for {symbol}.")

        stock["hist_vol"] = self._hist_vol(stock["close"], window=30)
        stock = stock.dropna(subset=["hist_vol"])
        stock = stock[stock.index >= pd.to_datetime(start_date)]

        opt = self.data.option_daily_close_quote(
            option_ticker, start_date, end_date)
        if opt.empty:
            raise ValueError(f"No option quotes returned for {option_ticker}.")

        df = stock.join(opt, how="inner").dropna(
            subset=["bid", "ask", "mid", "hist_vol", "close"])
        if len(df) < lookback + 5:
            raise ValueError("Insufficient aligned trading days found.")

        is_put = option_type.lower() == "put"

        strategy = ZScoreMispricingStrategy(
            z_entry_sell=z_entry_sell,
            z_entry_buy=z_entry_buy,
            z_exit_threshold=z_exit,
            lookback_period=lookback,
            max_hold_days=max_hold_days,
            max_trade_drawdown=max_trade_drawdown,
            put_skew_offset=put_skew_offset,
            capital_allocation=capital_allocation,
            stock_tc_bps=stock_tc_bps,
        )

        self.equity_curve.clear()
        self.trade_log.clear()

        for i, (dt, row) in enumerate(df.iterrows()):
            S = float(row["close"])
            sigma = float(row["hist_vol"])
            bid = float(row["bid"])
            ask = float(row["ask"])
            mid = float(row["mid"])

            days_to_expiry = (pd.to_datetime(end_date) - dt).days
            T_current = max(1.0, float(days_to_expiry)) / 252.0

            current_equity = self._calc_current_equity(strategy, mid, S)

            pricer = BinomialOptionPricer(
                S0=S, K=strike, T=T_current, r=risk_free_rate, sigma=sigma,
                steps=binomial_steps, option_type=option_type, style="american"
            )
            result = pricer.price()
            model_price = result["price"]
            delta = result["delta"]

            signal, sig_data = strategy.generate_signal(
                model_price, mid, delta, T_current, pd.Timestamp(dt), S, is_put
            )
            sig_data.update({"bid": bid, "ask": ask, "mid": mid})

            # Terminal Liquidation
            if i == len(df) - 1 and strategy.position is not None:
                signal = "CLOSE"

            trade = strategy.execute_trade(
                signal, sig_data, S, pd.Timestamp(dt), current_equity)
            if trade:
                self.trade_log.append(trade)

            post_equity = self._calc_current_equity(strategy, mid, S)
            self.equity_curve.append({"date": dt, "equity": post_equity})

        metrics = strategy.get_performance_metrics()
        metrics.update(self._portfolio_metrics())
        return metrics

    def _portfolio_metrics(self) -> Dict:
        """Calculates institutional risk and return metrics."""
        if not self.equity_curve:
            return {"portfolio_return": 0.0, "portfolio_sharpe": 0.0, "max_drawdown": 0.0}

        eq = pd.DataFrame(self.equity_curve).set_index("date")["equity"]
        total_return = (
            eq.iloc[-1] - self.initial_capital) / self.initial_capital

        rets = np.log(eq / eq.shift(1)).dropna()
        sharpe = float((rets.mean() / rets.std()) *
                       np.sqrt(252)) if rets.std() > 0 else 0.0

        running_max = eq.cummax()
        max_dd = float(((eq - running_max) / running_max).min())

        return {
            "portfolio_return": float(total_return),
            "portfolio_sharpe": sharpe,
            "max_drawdown": max_dd,
            "final_equity": float(eq.iloc[-1]),
        }

    def export_to_csv(self, filename_prefix: str, output_dir: str = "backtest_results") -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if self.trade_log:
            pd.DataFrame(self.trade_log).to_csv(os.path.join(
                output_dir, f"{filename_prefix}_trades.csv"), index=False)
        if self.equity_curve:
            pd.DataFrame(self.equity_curve).to_csv(os.path.join(
                output_dir, f"{filename_prefix}_equity.csv"), index=False)

    def plot_results(self, title: str = "Equity Curve") -> None:
        if not self.equity_curve:
            return

        eq = pd.DataFrame(self.equity_curve)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(eq["date"], eq["equity"],
                     label="Portfolio Equity", linewidth=1.5)
        axes[0].axhline(self.initial_capital, linestyle="--", color="grey")
        axes[0].set_title(title)
        axes[0].set_ylabel("Equity ($)")
        axes[0].grid(alpha=0.3)

        equity_s = eq.set_index("date")["equity"]
        dd = (equity_s - equity_s.cummax()) / equity_s.cummax() * 100

        axes[1].fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
