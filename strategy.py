from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

from market_data import MarketData
from pricer import BinomialOptionPricer, implied_vol_crr, crr_price_given_sigma
from universe import BucketSpec, OptionUniverseBuilder
from surface import fit_surface, predict_surface_iv


@dataclass
class StrategyConfig:
    entry_price_edge: float = 0.20
    entry_iv_edge: float = 0.015
    exit_iv_edge: float = 0.005
    max_holding_days: int = 10
    min_volume: int = 1
    max_open_positions: int = 2
    delta_rehedge_threshold_shares: float = 8.0
    min_chain_size: int = 6
    max_spread_frac: float = 0.40
    trade_rich_options: bool = True
    trade_cheap_options: bool = True


@dataclass
class Position:
    bucket: str
    option_ticker: str
    option_type: str
    side: int
    contracts: int
    entry_date: pd.Timestamp
    expiry: pd.Timestamp
    strike: float
    entry_option_price: float
    entry_spot: float
    hedge_shares: float
    entry_delta: float
    surface_resid_entry: float
    days_held: int = 0
    last_option_price: float = np.nan
    last_spot: float = np.nan


class CRRSurfaceDeltaHedgeStrategy:
    def __init__(
        self,
        api_key: str,
        bucket: BucketSpec,
        risk_free_rate: float,
        dividend_yield: float,
        crr_steps: int,
        assumed_spread_bps: float,
        underlying_symbol: str = "SPY",
        config: StrategyConfig | None = None,
    ):
        self.api_key = api_key
        self.bucket = bucket
        self.r = risk_free_rate
        self.q = dividend_yield
        self.steps = crr_steps
        self.assumed_spread_bps = assumed_spread_bps
        self.underlying_symbol = underlying_symbol
        self.cfg = config or StrategyConfig()

        self.md = MarketData(api_key=api_key)
        self.universe = OptionUniverseBuilder(
            api_key=api_key, underlying_symbol=underlying_symbol)

    def build_daily_chain_snapshot(self, date: pd.Timestamp, spot: float, contract_df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for _, row in contract_df.iterrows():
            ticker = row["option_ticker"]
            strike = float(row["strike"])
            expiry = pd.Timestamp(row["expiry"])
            option_type = str(row["contract_type"]).lower()

            if date >= expiry:
                continue

            opt = self.md.get_option_daily_aggs(
                option_ticker=ticker,
                start_date=date.strftime("%Y-%m-%d"),
                end_date=date.strftime("%Y-%m-%d"),
                assumed_spread_bps=self.assumed_spread_bps,
            )
            if opt.empty:
                continue

            mid = float(opt["mid"].iloc[0])
            spread = float(opt["spread"].iloc[0])
            volume = float(opt["volume"].iloc[0]) if pd.notna(
                opt["volume"].iloc[0]) else 0.0

            dte = int((expiry - date).days)
            if dte <= 0 or mid <= 0 or volume < self.cfg.min_volume:
                continue

            T = dte / 252.0
            iv = implied_vol_crr(
                market_price=mid,
                S0=spot,
                K=strike,
                T=T,
                r=self.r,
                q=self.q,
                steps=self.steps,
                option_type=option_type,
                style="american",
            )
            if not np.isfinite(iv):
                continue

            rows.append({
                "date": date,
                "bucket": self.bucket.label,
                "option_ticker": ticker,
                "option_type": option_type,
                "strike": strike,
                "expiry": expiry,
                "dte": dte,
                "T": T,
                "spot": spot,
                "market_mid": mid,
                "spread": spread,
                "volume": volume,
                "iv": iv,
                "moneyness": strike / spot,
                "log_moneyness": np.log(strike / spot),
            })

        return pd.DataFrame(rows)

    def enrich_with_surface(self, chain_df: pd.DataFrame) -> pd.DataFrame:
        if len(chain_df) < self.cfg.min_chain_size:
            return pd.DataFrame()

        beta = fit_surface(chain_df)
        out_rows = []

        for _, row in chain_df.iterrows():
            fitted_iv = predict_surface_iv(
                beta, row["spot"], row["strike"], row["T"])

            fair_price = crr_price_given_sigma(
                S0=row["spot"],
                K=row["strike"],
                T=row["T"],
                r=self.r,
                q=self.q,
                sigma=fitted_iv,
                steps=self.steps,
                option_type=row["option_type"],
                style="american",
            )

            pricer = BinomialOptionPricer(
                S0=row["spot"],
                K=row["strike"],
                T=row["T"],
                r=self.r,
                q=self.q,
                sigma=fitted_iv,
                steps=self.steps,
                option_type=row["option_type"],
                style="american",
            )
            fair_delta = pricer.delta()

            residual_price = fair_price - row["market_mid"]
            residual_iv = fitted_iv - row["iv"]

            tmp = row.to_dict()
            tmp.update({
                "surface_iv": fitted_iv,
                "surface_fair_price": fair_price,
                "surface_delta": fair_delta,
                "residual_price": residual_price,
                "residual_iv": residual_iv,
                "abs_residual_price": abs(residual_price),
                "abs_residual_iv": abs(residual_iv),
            })
            out_rows.append(tmp)

        return pd.DataFrame(out_rows)

    def generate_candidates(self, surface_df: pd.DataFrame) -> pd.DataFrame:
        if surface_df.empty:
            return surface_df

        df = surface_df.copy()

        long_mask = (
            (df["residual_price"] >= self.cfg.entry_price_edge) &
            (df["residual_iv"] >= self.cfg.entry_iv_edge) &
            (df["spread"] < df["market_mid"] * self.cfg.max_spread_frac)
        )

        short_mask = (
            (df["residual_price"] <= -self.cfg.entry_price_edge) &
            (df["residual_iv"] <= -self.cfg.entry_iv_edge) &
            (df["spread"] < df["market_mid"] * self.cfg.max_spread_frac)
        )

        frames = []

        if self.cfg.trade_cheap_options:
            longs = df.loc[long_mask].copy()
            longs["signal_side"] = 1
            frames.append(longs)

        if self.cfg.trade_rich_options:
            shorts = df.loc[short_mask].copy()
            shorts["signal_side"] = -1
            frames.append(shorts)

        if not frames:
            return pd.DataFrame()

        cands = pd.concat(frames, ignore_index=True)
        if cands.empty:
            return cands

        cands["signal_strength"] = cands["abs_residual_price"] + \
            10.0 * cands["abs_residual_iv"]
        cands = cands.sort_values("signal_strength", ascending=False)
        return cands

    def run_backtest(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        und = self.md.get_underlying_daily(
            self.underlying_symbol, start_date, end_date)
        if und.empty:
            raise RuntimeError(
                f"No underlying data for {self.underlying_symbol}.")

        positions: List[Position] = []
        trade_rows: List[Dict] = []
        equity_rows: List[Dict] = []
        signal_rows: List[Dict] = []

        all_dates = list(und.index)

        for date in all_dates:
            spot = float(und.loc[date, "close"])
            daily_pnl = 0.0
            to_close = []

            for idx, pos in enumerate(positions):
                if date >= pos.expiry:
                    to_close.append(idx)
                    continue

                opt_df = self.md.get_option_daily_aggs(
                    option_ticker=pos.option_ticker,
                    start_date=date.strftime("%Y-%m-%d"),
                    end_date=date.strftime("%Y-%m-%d"),
                    assumed_spread_bps=self.assumed_spread_bps,
                )
                if opt_df.empty:
                    continue

                option_mid = float(opt_df["mid"].iloc[0])

                option_pnl = 0.0
                hedge_pnl = 0.0

                if np.isfinite(pos.last_option_price):
                    option_pnl = pos.side * pos.contracts * \
                        100.0 * (option_mid - pos.last_option_price)

                if np.isfinite(pos.last_spot):
                    hedge_pnl = pos.hedge_shares * (spot - pos.last_spot)

                daily_pnl += option_pnl + hedge_pnl

                pos.last_option_price = option_mid
                pos.last_spot = spot
                pos.days_held += 1

                T = max((pos.expiry - date).days / 252.0, 1 / 252.0)
                current_iv = implied_vol_crr(
                    market_price=option_mid,
                    S0=spot,
                    K=pos.strike,
                    T=T,
                    r=self.r,
                    q=self.q,
                    steps=self.steps,
                    option_type=pos.option_type,
                    style="american",
                )

                if np.isfinite(current_iv):
                    pricer = BinomialOptionPricer(
                        S0=spot,
                        K=pos.strike,
                        T=T,
                        r=self.r,
                        q=self.q,
                        sigma=current_iv,
                        steps=self.steps,
                        option_type=pos.option_type,
                        style="american",
                    )
                    new_delta = pricer.delta()
                    desired_hedge = -pos.side * pos.contracts * 100.0 * new_delta
                    if abs(desired_hedge - pos.hedge_shares) > self.cfg.delta_rehedge_threshold_shares:
                        pos.hedge_shares = desired_hedge

                if (pos.expiry - date).days < 7 or pos.days_held >= self.cfg.max_holding_days:
                    trade_rows.append({
                        "date": date,
                        "bucket": pos.bucket,
                        "action": "EXIT",
                        "option_ticker": pos.option_ticker,
                        "option_type": pos.option_type,
                        "side": pos.side,
                        "strike": pos.strike,
                        "expiry": pos.expiry,
                        "spot": spot,
                        "option_mid": option_mid,
                        "hedge_shares": pos.hedge_shares,
                        "days_held": pos.days_held,
                    })
                    to_close.append(idx)

            if to_close:
                positions = [p for i, p in enumerate(
                    positions) if i not in set(to_close)]

            contract_df = self.universe.list_contracts(
                date.strftime("%Y-%m-%d"), spot, self.bucket)

            if contract_df.empty:
                equity_rows.append({"date": date, "bucket": self.bucket.label, "spot": spot,
                                   "daily_pnl": daily_pnl, "open_positions": len(positions)})
                continue

            chain_df = self.build_daily_chain_snapshot(date, spot, contract_df)
            if chain_df.empty:
                equity_rows.append({"date": date, "bucket": self.bucket.label, "spot": spot,
                                   "daily_pnl": daily_pnl, "open_positions": len(positions)})
                continue

            surface_df = self.enrich_with_surface(chain_df)
            if surface_df.empty:
                equity_rows.append({"date": date, "bucket": self.bucket.label, "spot": spot,
                                   "daily_pnl": daily_pnl, "open_positions": len(positions)})
                continue

            signal_rows.extend(surface_df.to_dict("records"))
            cands = self.generate_candidates(surface_df)

            if not cands.empty and len(positions) < self.cfg.max_open_positions:
                already_open = {p.option_ticker for p in positions}

                for _, row in cands.iterrows():
                    if len(positions) >= self.cfg.max_open_positions:
                        break
                    if row["option_ticker"] in already_open:
                        continue

                    side = int(row["signal_side"])
                    contracts = 1
                    delta = float(row["surface_delta"])
                    hedge_shares = -side * contracts * 100.0 * delta

                    pos = Position(
                        bucket=self.bucket.label,
                        option_ticker=row["option_ticker"],
                        option_type=row["option_type"],
                        side=side,
                        contracts=contracts,
                        entry_date=date,
                        expiry=pd.Timestamp(row["expiry"]),
                        strike=float(row["strike"]),
                        entry_option_price=float(row["market_mid"]),
                        entry_spot=spot,
                        hedge_shares=hedge_shares,
                        entry_delta=delta,
                        surface_resid_entry=float(row["residual_price"]),
                        last_option_price=float(row["market_mid"]),
                        last_spot=spot,
                    )
                    positions.append(pos)

                    trade_rows.append({
                        "date": date,
                        "bucket": self.bucket.label,
                        "action": "ENTRY_LONG_CHEAP" if side == 1 else "ENTRY_SHORT_RICH",
                        "option_ticker": row["option_ticker"],
                        "option_type": row["option_type"],
                        "side": side,
                        "strike": row["strike"],
                        "expiry": row["expiry"],
                        "spot": spot,
                        "market_mid": row["market_mid"],
                        "surface_fair_price": row["surface_fair_price"],
                        "surface_iv": row["surface_iv"],
                        "market_iv": row["iv"],
                        "residual_price": row["residual_price"],
                        "residual_iv": row["residual_iv"],
                        "delta": delta,
                        "hedge_shares": hedge_shares,
                    })

            equity_rows.append({
                "date": date,
                "bucket": self.bucket.label,
                "spot": spot,
                "daily_pnl": daily_pnl,
                "open_positions": len(positions),
            })

        equity = pd.DataFrame(equity_rows).sort_values("date")
        trades = pd.DataFrame(trade_rows).sort_values(
            "date") if trade_rows else pd.DataFrame()
        signals = pd.DataFrame(signal_rows).sort_values(
            ["date", "option_ticker"]) if signal_rows else pd.DataFrame()

        if not equity.empty:
            equity["cum_pnl"] = equity["daily_pnl"].cumsum()

        return {
            "equity": equity,
            "trades": trades,
            "signals": signals,
        }
