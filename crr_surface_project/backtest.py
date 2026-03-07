"""
backtest.py
Thin orchestration layer for running a single-bucket backtest.

This module exposes ``run_bucket_backtest``, a convenience function that
constructs a ``CRRSurfaceDeltaHedgeStrategy`` from the supplied parameters
and delegates execution to its ``run_backtest`` method.

Keeping orchestration separate from the strategy class itself makes it easy
to call from both the interactive research scripts (``run_research.py``,
``run_full_backtest.py``) and the parallel sensitivity sweep
(``run_sensitivity_study.py``).
"""

from __future__ import annotations

from typing import Dict, Optional

from strategy import CRRSurfaceDeltaHedgeStrategy, StrategyConfig
from universe import BucketSpec


def run_bucket_backtest(
    api_key: str,
    bucket: BucketSpec,
    start_date: str,
    end_date: str,
    risk_free_rate: float,
    dividend_yield: float,
    crr_steps: int,
    assumed_spread_bps: float,
    strategy_config: Optional[StrategyConfig] = None,
    market_data_provider=None,
    universe_builder=None,
) -> Dict:
    """
    Construct and run a ``CRRSurfaceDeltaHedgeStrategy`` for one bucket.

    Parameters
    ----------
    api_key : str
        Polygon.io API key. Only used when ``market_data_provider`` and
        ``universe_builder`` are ``None`` (i.e. live-API mode).
    bucket : BucketSpec
        Option universe slice to trade (e.g. ``BEST_BUCKET``).
    start_date : str
        Inclusive backtest start date in ``YYYY-MM-DD`` format.
    end_date : str
        Inclusive backtest end date in ``YYYY-MM-DD`` format.
    risk_free_rate : float
        Continuously compounded risk-free rate (annualised).
    dividend_yield : float
        Continuous dividend yield of the underlying (annualised).
    crr_steps : int
        Number of CRR binomial tree steps.
    assumed_spread_bps : float
        Synthetic bid-ask spread in basis points applied to closing prices.
    strategy_config : StrategyConfig, optional
        Strategy hyperparameters. Defaults to ``StrategyConfig()`` if ``None``.
    market_data_provider : optional
        Injectable market data object (e.g. ``CachedMarketData``). Defaults
        to a live ``MarketData`` instance when ``None``.
    universe_builder : optional
        Injectable contract universe object (e.g. ``CachedUniverseBuilder``).
        Defaults to a live ``OptionUniverseBuilder`` instance when ``None``.

    Returns
    -------
    dict
        Output from ``CRRSurfaceDeltaHedgeStrategy.run_backtest()``:
        ``{"equity": DataFrame, "trades": DataFrame, "signals": DataFrame}``.
    """
    strat = CRRSurfaceDeltaHedgeStrategy(
        api_key=api_key,
        bucket=bucket,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        crr_steps=crr_steps,
        assumed_spread_bps=assumed_spread_bps,
        config=strategy_config,
        market_data_provider=market_data_provider,
        universe_builder=universe_builder,
    )
    return strat.run_backtest(start_date=start_date, end_date=end_date)
