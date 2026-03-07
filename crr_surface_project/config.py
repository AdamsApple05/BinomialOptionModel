"""
config.py
Global configuration for the CRR Surface Delta-Hedge Strategy.

All runtime parameters are centralised here. Override defaults by setting
environment variables or by passing keyword arguments when constructing
``GlobalConfig``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalConfig:
    """
    Immutable configuration container for the entire research framework.

    Attributes
    ----------
    api_key : str
        Polygon.io REST API key. Reads from the ``POLYGON_API_KEY`` environment
        variable by default. Required for any live or download step.
    underlying_symbol : str
        Ticker symbol of the underlying equity. Defaults to ``"SPY"``.
    risk_free_rate : float
        Continuously compounded risk-free rate (annualised). Used as the
        drift term in the CRR binomial tree. Defaults to ``0.045`` (4.5 %).
    dividend_yield : float
        Continuous dividend yield of the underlying (annualised). Subtracted
        from the drift in the risk-neutral measure. Defaults to ``0.012``
        (1.2 % — approximate SPY historical yield).
    crr_steps : int
        Number of time steps in the CRR binomial tree. Higher values reduce
        discretisation error at the cost of O(N²) runtime. Defaults to ``80``;
        see ``run_convergence.py`` for the accuracy–speed trade-off analysis.
    assumed_spread_bps : float
        Synthetic bid-ask half-spread in basis points. Applied symmetrically
        around the Polygon OHLC close price to produce bid and ask quotes.
        Defaults to ``25.0`` bps (0.25 %).
    trading_days_per_year : int
        Business-day convention used for annualising volatility and Sharpe
        ratios. Defaults to ``252``.
    start_date : str
        Backtest start date in ``YYYY-MM-DD`` format. Defaults to
        ``"2024-01-01"``.
    end_date : str
        Backtest end date in ``YYYY-MM-DD`` format. Defaults to
        ``"2024-04-01"``.
    cache_dir : str
        Path to the local CSV data cache directory populated by
        ``run_download.py``. Resolved relative to the current working
        directory. Defaults to ``"data"``.
    calls_per_minute : int
        Maximum Polygon API calls per minute. Set to ``5`` for the free tier
        and ``1000`` (effectively unlimited) for paid plans. Defaults to
        ``1000``.
    """

    # ── API ──────────────────────────────────────────────────────────────────
    api_key: str = os.getenv("POLYGON_API_KEY", "YOUR_API_KEY_HERE")
    underlying_symbol: str = "SPY"

    # ── Pricing model ────────────────────────────────────────────────────────
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.012
    crr_steps: int = 80

    # ── Market data assumptions ──────────────────────────────────────────────
    assumed_spread_bps: float = 25.0
    trading_days_per_year: int = 252

    # ── Backtest window ──────────────────────────────────────────────────────
    start_date: str = "2024-01-01"
    end_date: str = "2024-04-01"

    # ── Local data cache ─────────────────────────────────────────────────────
    cache_dir: str = "data"
    calls_per_minute: int = 1000
