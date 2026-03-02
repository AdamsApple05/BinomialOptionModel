"""
zscore_strategy.py
------------------
Core logic for the Delta-Neutral Volatility Arbitrage strategy.
Handles signal generation, position sizing, dynamic delta hedging, and risk management.
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Optional, Tuple


class ZScoreMispricingStrategy:
    """
    Evaluates option mispricing against a theoretical binomial model using a rolling Z-Score.
    Includes skew-adjusted signals, dynamic delta hedging bands, and notional margin limits.
    """

    def __init__(
        self,
        z_entry_sell: float = 1.25,
        z_entry_buy: float = 2.25,
        z_exit_threshold: float = 0.5,
        lookback_period: int = 15,
        max_hold_days: int = 21,
        max_trade_drawdown: float = -0.15,
        capital_allocation: float = 0.95,
        put_skew_offset: float = 0.5,
        stock_tc_bps: float = 0.0,
    ):
        self.z_entry_sell = z_entry_sell
        self.z_entry_buy = z_entry_buy
        self.z_exit = z_exit_threshold
        self.lookback = lookback_period
        self.max_hold_days = max_hold_days
        self.max_trade_drawdown = max_trade_drawdown
        self.capital_allocation = capital_allocation
        self.put_skew_offset = put_skew_offset
        self.stock_tc_bps = stock_tc_bps

        self.mispricing_history: deque = deque(maxlen=lookback_period)
        self.position: Optional[Dict] = None
        self.pnl_history: list = []
        self.trades: list = []

    def calculate_zscore(self, current_mispricing: float) -> float:
        if len(self.mispricing_history) < self.lookback:
            return 0.0
        arr = np.array(self.mispricing_history, dtype=float)
        std = arr.std()
        return float((current_mispricing - arr.mean()) / std) if std != 0 else 0.0

    def generate_signal(
        self,
        model_price: float,
        market_mid: float,
        current_delta: float,
        ttm: float,
        current_date: pd.Timestamp,
        stock_price: float,
        is_put: bool
    ) -> Tuple[str, Dict]:

        # Adjust for natural market volatility skew to prevent false signals on puts
        mispricing = market_mid - model_price
        effective_mispricing = mispricing - \
            (self.put_skew_offset if is_put else 0)

        z = self.calculate_zscore(effective_mispricing)
        self.mispricing_history.append(effective_mispricing)

        data = {
            "model_price": float(model_price),
            "market_mid": float(market_mid),
            "zscore": float(z),
            "delta": float(current_delta)
        }

        if self.position is None:
            if z > self.z_entry_sell:
                return "SELL", data
            if z < -self.z_entry_buy:
                return "BUY", data
            return "HOLD", data

        days_held = (current_date - self.position["entry_timestamp"]).days
        contracts = abs(self.position["contracts"])
        opt_val = market_mid * 100.0 * contracts
        stk_val = float(self.position["hedge_shares"]) * stock_price

        if self.position["type"] == "short_option":
            current_mtm = float(
                self.position["initial_proceeds"]) - opt_val + stk_val
            initial_cap = abs(self.position["initial_proceeds"])
        else:
            current_mtm = opt_val - \
                float(self.position["initial_cost"]) + stk_val
            initial_cap = abs(self.position["initial_cost"])

        # Hard stop-loss limit
        if initial_cap > 0 and (current_mtm / initial_cap) < self.max_trade_drawdown:
            return "CLOSE", data

        # Time stop or mean-reversion exit
        if days_held >= self.max_hold_days or abs(z) < self.z_exit:
            return "CLOSE", data

        # Dynamic delta hedging bands (tighten as TTM increases)
        dynamic_threshold = 0.05 + (0.15 * (1.0 - ttm))
        if abs(current_delta - self.position["delta"]) > dynamic_threshold:
            return "REBALANCE", data

        return "HOLD", data

    def execute_trade(
        self,
        signal: str,
        signal_data: Dict,
        stock_price: float,
        timestamp: pd.Timestamp,
        current_equity: float,
    ) -> Optional[Dict]:

        if signal == "HOLD":
            return None

        bid, ask, mid, delta = map(float, [
            signal_data["bid"], signal_data["ask"], signal_data["mid"], signal_data["delta"]
        ])

        trade = {
            "timestamp": timestamp, "signal": signal, "stock_price": stock_price,
            "model_price": signal_data["model_price"], "mid": mid, "bid": bid, "ask": ask,
            "zscore": signal_data["zscore"]
        }

        def stock_tc(amt: float) -> float:
            return abs(amt) * (self.stock_tc_bps / 10_000.0)

        if signal in ["SELL", "BUY"]:
            capital_to_deploy = current_equity * self.capital_allocation

            if signal == "SELL":
                net_cost = (abs(delta) * 100 * stock_price) - (bid * 100)
                entry_px = bid
            else:
                net_cost = (ask * 100) + (abs(delta) * 100 * stock_price)
                entry_px = ask

            # Apply 15% notional margin floor to prevent infinite sizing anomalies
            margin_floor = (stock_price * 100) * 0.15
            capital_per_contract = max(net_cost, margin_floor)
            contracts = max(1, int(capital_to_deploy / capital_per_contract))

        if signal == "SELL":
            hedge_shares = delta * 100 * contracts
            hedge_cost = hedge_shares * stock_price
            tc = stock_tc(hedge_cost)
            self.position = {
                "type": "short_option", "contracts": -contracts, "entry_price": entry_px,
                "hedge_shares": hedge_shares, "delta": delta, "entry_timestamp": timestamp,
                "initial_proceeds": (entry_px * 100 * contracts) - hedge_cost - tc,
            }
            trade.update({"contracts": -contracts, "entry_option_px": entry_px, "hedge_shares": hedge_shares,
                          "initial_proceeds": self.position["initial_proceeds"], "stock_tc": tc})

        elif signal == "BUY":
            hedge_shares = -delta * 100 * contracts
            hedge_proceeds = -hedge_shares * stock_price
            tc = stock_tc(hedge_proceeds)
            self.position = {
                "type": "long_option", "contracts": contracts, "entry_price": entry_px,
                "hedge_shares": hedge_shares, "delta": delta, "entry_timestamp": timestamp,
                "initial_cost": (entry_px * 100 * contracts) - hedge_proceeds + tc,
            }
            trade.update({"contracts": contracts, "entry_option_px": entry_px, "hedge_shares": hedge_shares,
                          "initial_cost": self.position["initial_cost"], "stock_tc": tc})

        elif signal == "CLOSE" and self.position:
            con = abs(self.position["contracts"])
            if self.position["type"] == "short_option":
                pnl = self.position["initial_proceeds"] - (ask * 100 * con) + (
                    self.position["hedge_shares"] * stock_price) - stock_tc(self.position["hedge_shares"] * stock_price)
            else:
                pnl = (bid * 100 * con) - self.position["initial_cost"] + (
                    self.position["hedge_shares"] * stock_price) - stock_tc(self.position["hedge_shares"] * stock_price)

            trade.update({"pnl": float(pnl), "entry_price": self.position["entry_price"], "exit_price": bid if self.position[
                         "type"] == "long_option" else ask, "hold_days": (timestamp - self.position["entry_timestamp"]).days})
            self.pnl_history.append(float(pnl))
            self.position = None

        elif signal == "REBALANCE" and self.position:
            mult = 1 if self.position["type"] == "short_option" else -1
            shares_to_trade = (
                mult * delta * 100 * abs(self.position["contracts"])) - self.position["hedge_shares"]
            tc = stock_tc(shares_to_trade * stock_price)

            self.position["hedge_shares"] += shares_to_trade
            self.position["delta"] = delta

            if self.position["type"] == "short_option":
                self.position["initial_proceeds"] -= (
                    shares_to_trade * stock_price) + tc
            else:
                self.position["initial_cost"] += (
                    shares_to_trade * stock_price) + tc

            trade.update({"shares_traded": float(shares_to_trade), "cost": float(
                shares_to_trade * stock_price), "stock_tc": tc, "new_delta": delta})

        self.trades.append(trade)
        return trade

    def get_performance_metrics(self) -> Dict:
        if not self.pnl_history:
            return {"total_pnl": 0.0, "num_trades": 0, "win_rate": 0.0}

        pnl = np.array(self.pnl_history, dtype=float)
        return {
            "total_pnl": float(pnl.sum()),
            "num_trades": len(pnl),
            "win_rate": float((pnl > 0).sum() / len(pnl)),
            "avg_pnl": float(pnl.mean()),
            "sharpe_ratio": float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0,
            "max_win": float(pnl.max()),
            "max_loss": float(pnl.min())
        }
