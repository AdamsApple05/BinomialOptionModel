"""
strategy.py
Core strategy logic for the CRR Surface Delta-Hedge Strategy.

Architecture
------------
``CRRSurfaceDeltaHedgeStrategy`` is the central class. Its ``run_backtest``
method iterates over each trading day in the requested date range and:

1. **Marks open positions to market** — fetches each held contract's closing
   price and computes daily option P&L and hedge P&L.
2. **Re-hedges delta** — recomputes CRR delta for each open position and
   adjusts the underlying share hedge when the hedge error exceeds the
   rebalancing threshold.
3. **Checks exit conditions** — closes positions that have exceeded the
   maximum holding period or are within the near-expiry buffer.
4. **Builds a daily chain snapshot** — fetches market data for every
   contract in the bucket universe and computes market-implied volatilities
   via bisection.
5. **Fits the IV surface** — runs cross-sectional OLS regression on the
   chain to produce a daily surface model.
6. **Generates signals** — computes CRR fair prices at surface-implied IVs,
   identifies price and IV residuals, and ranks trading opportunities.
7. **Enters new positions** — selects the top-ranked candidates subject to
   the maximum open-positions constraint and enters with a delta-neutral
   initial hedge.

The strategy supports dependency injection of the data provider
(``market_data_provider``) and contract universe builder
(``universe_builder``), which allows seamless switching between the live
Polygon API (``MarketData``, ``OptionUniverseBuilder``) and the local CSV
cache (``CachedMarketData``, ``CachedUniverseBuilder``) without changing
any strategy logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from market_data import MarketData
from pricer import BinomialOptionPricer, crr_price_given_sigma, implied_vol_crr
from surface import fit_surface, predict_surface_iv
from universe import BucketSpec, OptionUniverseBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """
    Tunable hyperparameters for signal generation, position sizing, and
    risk management.

    Attributes
    ----------
    entry_price_edge : float
        Minimum absolute price residual (CRR fair − market mid) required to
        enter a position. Acts as a price-based signal filter. Defaults to
        ``0.20`` (20 cents per share, or $20 per contract).
    entry_iv_edge : float
        Minimum absolute IV residual (surface IV − market IV) required to
        enter a position. Acts as a vol-based signal filter. Defaults to
        ``0.015`` (1.5 vol points).
    exit_iv_edge : float
        Exit threshold: close the position when the IV residual falls below
        this level, indicating the edge has been realised. Defaults to
        ``0.005``.
    max_holding_days : int
        Hard maximum number of calendar days to hold any position.
        Defaults to ``10``.
    min_volume : int
        Minimum daily traded volume (contracts) required to consider a
        contract eligible for entry. Defaults to ``1``.
    max_open_positions : int
        Maximum number of simultaneously open positions. Defaults to ``2``.
    delta_rehedge_threshold_shares : float
        Minimum absolute delta-hedge error in shares before triggering a
        rebalance trade. Defaults to ``8.0`` shares.
    min_chain_size : int
        Minimum number of contracts in the daily chain required to fit a
        stable IV surface. Defaults to ``6``.
    max_spread_frac : float
        Maximum allowable bid-ask spread as a fraction of the option mid
        price. Contracts with wider spreads are excluded from the signal
        universe. Defaults to ``0.40`` (40 % of mid).
    trade_rich_options : bool
        If ``True``, generate short (sell) signals on rich options
        (negative residual). Defaults to ``True``.
    trade_cheap_options : bool
        If ``True``, generate long (buy) signals on cheap options
        (positive residual). Defaults to ``True``.
    """

    entry_price_edge: float = 0.05
    entry_iv_edge: float = 0.001
    exit_iv_edge: float = 0.005
    max_holding_days: int = 10
    min_volume: int = 1
    max_open_positions: int = 2
    delta_rehedge_threshold_shares: float = 8.0
    min_chain_size: int = 6
    max_spread_frac: float = 0.40
    trade_rich_options: bool = True
    trade_cheap_options: bool = True
    dd_min_positions: int = 1
    max_contracts: int = 3
    use_signal_strength_sizing: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Position state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """
    Runtime state for a single open option position.

    All monetary values are in dollars. ``side`` is ``+1`` (long) or
    ``-1`` (short). ``hedge_shares`` is negative for long options (sold
    underlying to hedge positive delta) and positive for short options.
    """

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


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

class CRRSurfaceDeltaHedgeStrategy:
    """
    CRR binomial surface relative-value strategy with delta hedging.

    Parameters
    ----------
    api_key : str
        Polygon.io API key (used only when ``market_data_provider`` and
        ``universe_builder`` are ``None``).
    bucket : BucketSpec
        Option universe slice to trade.
    risk_free_rate : float
        Continuously compounded risk-free rate (annualised).
    dividend_yield : float
        Continuous dividend yield of the underlying (annualised).
    crr_steps : int
        Number of CRR binomial tree steps for pricing and IV inversion.
    assumed_spread_bps : float
        Synthetic bid-ask spread in basis points applied to Polygon OHLC
        close prices.
    underlying_symbol : str
        Equity ticker. Defaults to ``"SPY"``.
    config : StrategyConfig, optional
        Strategy hyperparameters. Defaults to ``StrategyConfig()`` if ``None``.
    market_data_provider : optional
        Injectable market data object. Must expose ``get_underlying_daily``
        and ``get_option_daily_aggs`` with the same signature as
        ``MarketData``. Defaults to a live ``MarketData`` instance.
    universe_builder : optional
        Injectable contract universe object. Must expose ``list_contracts``
        with the same signature as ``OptionUniverseBuilder``. Defaults to a
        live ``OptionUniverseBuilder`` instance.
    """

    def __init__(
        self,
        api_key: str,
        bucket: BucketSpec,
        risk_free_rate: float,
        dividend_yield: float,
        crr_steps: int,
        assumed_spread_bps: float,
        underlying_symbol: str = "SPY",
        config: Optional[StrategyConfig] = None,
        market_data_provider=None,
        universe_builder=None,
    ) -> None:
        self.api_key = api_key
        self.bucket = bucket
        self.r = risk_free_rate
        self.q = dividend_yield
        self.steps = crr_steps
        self.assumed_spread_bps = assumed_spread_bps
        self.underlying_symbol = underlying_symbol
        self.cfg = config or StrategyConfig()

        self.md = (
            market_data_provider
            if market_data_provider is not None
            else MarketData(api_key=api_key)
        )
        self.universe = (
            universe_builder
            if universe_builder is not None
            else OptionUniverseBuilder(api_key=api_key, underlying_symbol=underlying_symbol)
        )

    # ── Chain snapshot ────────────────────────────────────────────────────────

    def build_daily_chain_snapshot(
        self,
        date: pd.Timestamp,
        spot: float,
        contract_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fetch market data for each contract in ``contract_df`` and compute
        market-implied volatilities via bisection.

        Contracts are excluded from the snapshot if they:
        * have already expired (``date >= expiry``),
        * have zero or negative mid price,
        * have daily volume below ``cfg.min_volume``, or
        * yield a non-finite implied volatility from the bisection solver.

        Parameters
        ----------
        date : pd.Timestamp
            Current simulation date.
        spot : float
            Underlying closing price on ``date``.
        contract_df : pd.DataFrame
            Contract universe from ``universe.list_contracts``.

        Returns
        -------
        pd.DataFrame
            One row per valid contract. Columns include ``"iv"``,
            ``"log_moneyness"``, ``"T"``, and all original contract fields.
        """
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
                S0=spot, K=strike, T=T,
                r=self.r, q=self.q,
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

    # ── Surface enrichment ────────────────────────────────────────────────────

    def enrich_with_surface(self, chain_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the IV surface to ``chain_df`` and compute CRR fair values and
        residuals for every contract.

        Returns an empty DataFrame if the chain is too small to fit a stable
        surface (fewer than ``cfg.min_chain_size`` contracts).

        Parameters
        ----------
        chain_df : pd.DataFrame
            Output from ``build_daily_chain_snapshot``.

        Returns
        -------
        pd.DataFrame
            ``chain_df`` enriched with ``surface_iv``, ``surface_fair_price``,
            ``surface_delta``, ``residual_price``, ``residual_iv``,
            ``abs_residual_price``, and ``abs_residual_iv``.
        """
        if len(chain_df) < self.cfg.min_chain_size:
            return pd.DataFrame()

        beta = fit_surface(chain_df)
        out_rows = []

        for _, row in chain_df.iterrows():
            fitted_iv = predict_surface_iv(
                beta, row["spot"], row["strike"], row["T"])

            fair_price = crr_price_given_sigma(
                S0=row["spot"], K=row["strike"], T=row["T"],
                r=self.r, q=self.q,
                sigma=fitted_iv, steps=self.steps,
                option_type=row["option_type"], style="american",
            )

            fair_delta = BinomialOptionPricer(
                S0=row["spot"], K=row["strike"], T=row["T"],
                r=self.r, q=self.q,
                sigma=fitted_iv, steps=self.steps,
                option_type=row["option_type"], style="american",
            ).delta()

            tmp = row.to_dict()
            tmp.update({
                "surface_iv": fitted_iv,
                "surface_fair_price": fair_price,
                "surface_delta": fair_delta,
                "residual_price": fair_price - row["market_mid"],
                "residual_iv": fitted_iv - row["iv"],
                "abs_residual_price": abs(fair_price - row["market_mid"]),
                "abs_residual_iv": abs(fitted_iv - row["iv"]),
            })
            out_rows.append(tmp)

        return pd.DataFrame(out_rows)

    # ── Signal generation ─────────────────────────────────────────────────────

    def generate_candidates(self, surface_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter ``surface_df`` down to actionable entry candidates and rank
        them by signal strength.

        A **long** (buy cheap) signal requires:
        * ``residual_price >= entry_price_edge``
        * ``residual_iv >= entry_iv_edge``
        * ``spread < market_mid * max_spread_frac``

        A **short** (sell rich) signal requires:
        * ``residual_price <= -entry_price_edge``
        * ``residual_iv <= -entry_iv_edge``
        * ``spread < market_mid * max_spread_frac``

        Signal strength is defined as::

            signal_strength = |residual_price| + 10 * |residual_iv|

        Parameters
        ----------
        surface_df : pd.DataFrame
            Output from ``enrich_with_surface``.

        Returns
        -------
        pd.DataFrame
            Filtered and sorted candidates with a ``signal_side`` column
            (``+1`` = long, ``-1`` = short) and ``signal_strength``.
        """
        if surface_df.empty:
            return surface_df

        df = surface_df.copy()

        long_mask = (
            (df["residual_price"] >= self.cfg.entry_price_edge)
            & (df["residual_iv"] >= self.cfg.entry_iv_edge)
            & (df["spread"] < df["market_mid"] * self.cfg.max_spread_frac)
        )

        short_mask = (
            (df["residual_price"] <= -self.cfg.entry_price_edge)
            & (df["residual_iv"] <= -self.cfg.entry_iv_edge)
            & (df["spread"] < df["market_mid"] * self.cfg.max_spread_frac)
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
        return cands.sort_values("signal_strength", ascending=False)

    # ── Position sizing helpers ───────────────────────────────────────────────

    def _scaled_max_positions(self, cum_pnl_history: list) -> int:
        """
        Reduce max open positions when current drawdown is a large fraction of
        the historical max drawdown.  Scales linearly from max_open_positions
        (at zero drawdown) down to dd_min_positions (at 100% of max drawdown).
        """
        if len(cum_pnl_history) < 10:
            return self.cfg.max_open_positions
        arr = np.array(cum_pnl_history, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd_series = arr - peak
        historical_max_dd = dd_series.min()   # most negative value
        if historical_max_dd >= 0:
            return self.cfg.max_open_positions
        current_dd = dd_series[-1]
        dd_ratio = current_dd / historical_max_dd   # 0 = no DD, 1 = at historical worst
        min_pos = self.cfg.dd_min_positions
        max_pos = self.cfg.max_open_positions
        scaled = max_pos - dd_ratio * (max_pos - min_pos)
        return max(min_pos, int(round(scaled)))

    def _signal_contracts(self, signal_strength: float, candidates_df) -> int:
        """
        Size contracts proportional to signal strength relative to the average
        signal on this day, capped at max_contracts.
        """
        if not self.cfg.use_signal_strength_sizing or candidates_df.empty:
            return 1
        avg_strength = float(candidates_df["signal_strength"].mean())
        if avg_strength <= 0:
            return 1
        ratio = signal_strength / avg_strength
        # ratio=1 -> 1 contract, ratio=2 -> 2 contracts, capped at max_contracts
        return max(1, min(int(round(ratio)), self.cfg.max_contracts))

    # ── Main backtest loop ────────────────────────────────────────────────────

    def run_backtest(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Run the full delta-hedge backtest over the specified date range.

        On each trading day the loop:

        1. Marks open positions to market (option P&L + hedge P&L).
        2. Re-hedges delta for positions whose hedge error exceeds the
           rebalancing threshold.
        3. Closes positions that have expired, exceeded ``max_holding_days``,
           or are within 7 DTE.
        4. Fetches the daily contract universe and builds a chain snapshot.
        5. Fits the IV surface and enriches contracts with CRR fair values.
        6. Checks IV-edge exit condition against today's surface residuals
           (only when ``exit_iv_edge > 0``).
        7. Generates entry candidates and opens new positions subject to
           the position limit.

        Parameters
        ----------
        start_date : str
            Inclusive backtest start date in ``YYYY-MM-DD`` format.
        end_date : str
            Inclusive backtest end date in ``YYYY-MM-DD`` format.

        Returns
        -------
        dict
            Keys:

            * ``"equity"`` — daily equity curve (date, bucket, spot,
              daily_pnl, open_positions, cum_pnl).
            * ``"trades"`` — transaction log with entry and exit records.
            * ``"signals"`` — full daily signal database (all contracts
              that passed surface enrichment, with residuals).

        Raises
        ------
        RuntimeError
            If no underlying price data is returned for the requested period.
        """
        und = self.md.get_underlying_daily(
            self.underlying_symbol, start_date, end_date)
        if und.empty:
            raise RuntimeError(
                f"No underlying data for {self.underlying_symbol}.")

        positions: List[Position] = []
        trade_rows: List[Dict] = []
        equity_rows: List[Dict] = []
        signal_rows: List[Dict] = []
        cum_pnl_history: List[float] = []
        running_cum_pnl: float = 0.0

        for date in tqdm(list(und.index), desc=f"Steps={self.steps}", leave=False):
            spot = float(und.loc[date, "close"])
            daily_pnl = 0.0
            to_close: List[int] = []

            # ── Mark-to-market open positions ─────────────────────────────
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
                    # No price data today — still age the position and check
                    # time-based exits so the position doesn't stay open forever.
                    pos.days_held += 1
                    if (pos.expiry - date).days < 7 or pos.days_held >= self.cfg.max_holding_days:
                        trade_rows.append({
                            "date": date,
                            "bucket": pos.bucket,
                            "action": "EXIT",
                            "option_ticker": pos.option_ticker,
                            "option_type": pos.option_type,
                            "side": pos.side,
                            "contracts": pos.contracts,
                            "strike": pos.strike,
                            "expiry": pos.expiry,
                            "spot": spot,
                            "option_mid": pos.last_option_price,
                            "hedge_shares": pos.hedge_shares,
                            "days_held": pos.days_held,
                        })
                        to_close.append(idx)
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

                # ── Delta re-hedge ────────────────────────────────────────
                T = max((pos.expiry - date).days / 252.0, 1.0 / 252.0)
                current_iv = implied_vol_crr(
                    market_price=option_mid,
                    S0=spot, K=pos.strike, T=T,
                    r=self.r, q=self.q,
                    steps=self.steps,
                    option_type=pos.option_type,
                    style="american",
                )

                if np.isfinite(current_iv):
                    new_delta = BinomialOptionPricer(
                        S0=spot, K=pos.strike, T=T,
                        r=self.r, q=self.q,
                        sigma=current_iv, steps=self.steps,
                        option_type=pos.option_type, style="american",
                    ).delta()
                    desired_hedge = -pos.side * pos.contracts * 100.0 * new_delta
                    if abs(desired_hedge - pos.hedge_shares) > self.cfg.delta_rehedge_threshold_shares:
                        pos.hedge_shares = desired_hedge

                # ── Exit conditions ───────────────────────────────────────
                # NOTE: exit_iv_edge is not checked here because the surface
                # is not fitted until later in the daily loop. IV-based exit
                # is handled in the post-surface pass below.
                if (pos.expiry - date).days < 7 or pos.days_held >= self.cfg.max_holding_days:
                    trade_rows.append({
                        "date": date,
                        "bucket": pos.bucket,
                        "action": "EXIT",
                        "option_ticker": pos.option_ticker,
                        "option_type": pos.option_type,
                        "side": pos.side,
                        "contracts": pos.contracts,
                        "strike": pos.strike,
                        "expiry": pos.expiry,
                        "spot": spot,
                        "option_mid": option_mid,
                        "hedge_shares": pos.hedge_shares,
                        "days_held": pos.days_held,
                    })
                    to_close.append(idx)

            if to_close:
                close_set = set(to_close)
                positions = [p for i, p in enumerate(
                    positions) if i not in close_set]

            # ── Build daily chain snapshot ────────────────────────────────
            contract_df = self.universe.list_contracts(
                date.strftime("%Y-%m-%d"), spot, self.bucket
            )

            if contract_df.empty:
                equity_rows.append({
                    "date": date, "bucket": self.bucket.label, "spot": spot,
                    "daily_pnl": daily_pnl, "open_positions": len(positions),
                })
                continue

            chain_df = self.build_daily_chain_snapshot(date, spot, contract_df)
            if chain_df.empty:
                equity_rows.append({
                    "date": date, "bucket": self.bucket.label, "spot": spot,
                    "daily_pnl": daily_pnl, "open_positions": len(positions),
                })
                continue

            # ── Fit surface and generate signals ──────────────────────────
            surface_df = self.enrich_with_surface(chain_df)
            if surface_df.empty:
                equity_rows.append({
                    "date": date, "bucket": self.bucket.label, "spot": spot,
                    "daily_pnl": daily_pnl, "open_positions": len(positions),
                })
                continue

            signal_rows.extend(surface_df.to_dict("records"))
            cands = self.generate_candidates(surface_df)

            # ── IV-edge exit pass (requires fitted surface) ───────────────
            # Check exit_iv_edge now that we have today's surface residuals.
            # Build a lookup: option_ticker -> current |residual_iv| from surface_df
            if self.cfg.exit_iv_edge > 0 and not surface_df.empty:
                iv_resid_map = surface_df.set_index("option_ticker")["residual_iv"].abs().to_dict()
                iv_to_close = []
                for idx, pos in enumerate(positions):
                    resid = iv_resid_map.get(pos.option_ticker)
                    if resid is not None and np.isfinite(resid) and resid < self.cfg.exit_iv_edge:
                        trade_rows.append({
                            "date": date,
                            "bucket": pos.bucket,
                            "action": "EXIT",
                            "option_ticker": pos.option_ticker,
                            "option_type": pos.option_type,
                            "side": pos.side,
                            "contracts": pos.contracts,
                            "strike": pos.strike,
                            "expiry": pos.expiry,
                            "spot": spot,
                            "option_mid": pos.last_option_price,
                            "hedge_shares": pos.hedge_shares,
                            "days_held": pos.days_held,
                        })
                        iv_to_close.append(idx)
                if iv_to_close:
                    close_set = set(iv_to_close)
                    positions = [p for i, p in enumerate(positions) if i not in close_set]

            # ── Enter new positions ───────────────────────────────────────
            position_limit = self._scaled_max_positions(cum_pnl_history)
            if not cands.empty and len(positions) < position_limit:
                already_open = {p.option_ticker for p in positions}

                for _, row in cands.iterrows():
                    if len(positions) >= position_limit:
                        break
                    if row["option_ticker"] in already_open:
                        continue

                    side = int(row["signal_side"])
                    contracts = self._signal_contracts(
                        float(row["signal_strength"]), cands)
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
                        "contracts": contracts,
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

            running_cum_pnl += daily_pnl
            cum_pnl_history.append(running_cum_pnl)
            equity_rows.append({
                "date": date,
                "bucket": self.bucket.label,
                "spot": spot,
                "daily_pnl": daily_pnl,
                "open_positions": len(positions),
            })

        # ── Assemble output DataFrames ────────────────────────────────────
        equity = pd.DataFrame(equity_rows).sort_values("date")
        trades = (
            pd.DataFrame(trade_rows).sort_values("date")
            if trade_rows
            else pd.DataFrame()
        )
        signals = (
            pd.DataFrame(signal_rows).sort_values(["date", "option_ticker"])
            if signal_rows
            else pd.DataFrame()
        )

        if not equity.empty:
            equity["cum_pnl"] = equity["daily_pnl"].cumsum()

        return {"equity": equity, "trades": trades, "signals": signals}
