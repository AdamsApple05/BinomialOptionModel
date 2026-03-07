"""
surface.py
Cross-sectional implied volatility surface fitting.

The surface model is a polynomial in log-moneyness and the square-root of
time-to-expiry, fit by ordinary least squares on each trading day's option
chain:

    IV(K, T) = β₀ + β₁·x + β₂·x² + β₃·√T

where ``x = ln(K / S)`` is the log-moneyness of the contract.

This parameterisation captures the volatility smile (quadratic in x) and the
term structure (linear in √T) with a minimal four-parameter model that is
stable even on small cross-sections (minimum ~6 contracts recommended).

The fitted surface is used to:

1. Interpolate / extrapolate fair IVs for all contracts in the chain.
2. Re-price each contract via the CRR binomial model at its surface IV.
3. Compute residuals (CRR fair price − market price) as trading signals.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def fit_surface(chain_df: pd.DataFrame) -> np.ndarray:
    """
    Fit the cross-sectional IV surface to a single day's option chain.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Must contain columns ``"strike"``, ``"spot"``, ``"T"`` (time-to-expiry
        in years), and ``"iv"`` (implied volatility recovered from market
        prices via bisection). Each row represents one option contract.

    Returns
    -------
    np.ndarray
        Shape ``(4,)`` coefficient vector ``[β₀, β₁, β₂, β₃]`` from the OLS
        regression. Pass to ``predict_surface_iv`` for out-of-sample
        predictions.

    Notes
    -----
    The regression is solved via ``numpy.linalg.lstsq`` with ``rcond=None``,
    which uses the full SVD and is robust to near-collinear design matrices.
    """
    x = np.log(chain_df["strike"] / chain_df["spot"])
    t = np.sqrt(chain_df["T"])
    y = chain_df["iv"]

    X = np.column_stack([
        np.ones(len(chain_df)),  # β₀ — level
        x,                        # β₁ — skew
        x ** 2,                   # β₂ — curvature (smile)
        t,                        # β₃ — term structure
    ])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict_surface_iv(beta: np.ndarray, spot: float, strike: float, T: float) -> float:
    """
    Evaluate the fitted surface at a given (strike, expiry) coordinate.

    Parameters
    ----------
    beta : np.ndarray
        Coefficient vector returned by ``fit_surface``.
    spot : float
        Current underlying price (used to compute log-moneyness).
    strike : float
        Option strike price.
    T : float
        Time-to-expiry in years.

    Returns
    -------
    float
        Surface-predicted implied volatility, clamped to ``[0.01, 2.50]``
        to prevent numerical instabilities in downstream pricing calls.
    """
    x = math.log(strike / spot)
    z = np.array([1.0, x, x * x, math.sqrt(T)])
    iv = float(z @ beta)
    return max(0.01, min(2.50, iv))
