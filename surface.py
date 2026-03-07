from __future__ import annotations
import math
import numpy as np
import pandas as pd


def fit_surface(chain_df: pd.DataFrame) -> np.ndarray:
    """
    Same-day cross-sectional IV fit:
        iv = b0 + b1*x + b2*x^2 + b3*sqrt(T)
    where:
        x = ln(K/S)
    """
    x = np.log(chain_df["strike"] / chain_df["spot"])
    t = np.sqrt(chain_df["T"])
    y = chain_df["iv"]

    X = np.column_stack([
        np.ones(len(chain_df)),
        x,
        x**2,
        t,
    ])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict_surface_iv(beta: np.ndarray, spot: float, strike: float, T: float) -> float:
    x = math.log(strike / spot)
    z = np.array([1.0, x, x*x, math.sqrt(T)])
    iv = float(z @ beta)
    return max(0.01, min(2.50, iv))
