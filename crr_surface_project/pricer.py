"""
pricer.py
Cox-Ross-Rubinstein (CRR) binomial option pricing engine.

This module implements:

* ``fast_build_stock_tree``  — Numba JIT-compiled stock price lattice.
* ``fast_build_option_tree`` — Numba JIT-compiled option value lattice with
  support for both European and American exercise.
* ``BinomialOptionPricer``   — High-level class wrapping the Numba kernels;
  exposes ``price()`` and ``delta()``.
* ``crr_price_given_sigma``  — Functional convenience wrapper.
* ``implied_vol_crr``        — Bisection-based implied volatility solver.

The JIT-compiled kernels cache their compiled form between Python sessions
(``cache=True``), so the JIT overhead only occurs on the first run.

References
----------
Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified
approach. *Journal of Financial Economics*, 7(3), 229–263.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numba import njit


# ─────────────────────────────────────────────────────────────────────────────
# Numba-compiled lattice kernels
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def fast_build_stock_tree(S0: float, u: float, d: float, steps: int) -> np.ndarray:
    """
    Build the recombining stock-price lattice.

    The lattice is stored in an upper-triangular (steps+1) x (steps+1) matrix
    where ``tree[j, i]`` is the stock price at time step ``i`` after ``j``
    down-moves (and ``i - j`` up-moves).

    Parameters
    ----------
    S0 : float
        Current stock price.
    u : float
        Up-move factor per step, ``u = exp(sigma * sqrt(dt))``.
    d : float
        Down-move factor per step, ``d = 1 / u``.
    steps : int
        Number of time steps ``N``.

    Returns
    -------
    np.ndarray
        Shape ``(steps+1, steps+1)`` stock-price array.
    """
    tree = np.zeros((steps + 1, steps + 1))
    tree[0, 0] = S0
    for i in range(1, steps + 1):
        tree[0, i] = tree[0, i - 1] * u
        for j in range(1, i + 1):
            tree[j, i] = tree[j - 1, i - 1] * d
    return tree


@njit(cache=True)
def fast_build_option_tree(
    stock_tree: np.ndarray,
    K: float,
    r: float,
    dt: float,
    p: float,
    steps: int,
    is_call: bool,
    is_american: bool,
) -> np.ndarray:
    """
    Build the option-value lattice by backward induction.

    Terminal payoffs are computed at expiry (column ``steps``), then the tree
    is rolled back to today (column 0) discounting at ``r`` per step. For
    American options the value at each interior node is taken as the maximum
    of the hold value and the immediate exercise (intrinsic) value.

    Parameters
    ----------
    stock_tree : np.ndarray
        Stock-price lattice from ``fast_build_stock_tree``.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate (annualised).
    dt : float
        Length of one time step in years, ``T / N``.
    p : float
        Risk-neutral probability of an up-move, clipped to ``(0, 1)``.
    steps : int
        Number of time steps ``N``.
    is_call : bool
        ``True`` for a call option, ``False`` for a put.
    is_american : bool
        ``True`` for American exercise, ``False`` for European.

    Returns
    -------
    np.ndarray
        Option-value lattice with the same shape as ``stock_tree``.
        The fair value today is ``opt[0, 0]``.
    """
    opt = np.zeros_like(stock_tree)

    # Terminal payoffs
    for j in range(steps + 1):
        if is_call:
            opt[j, steps] = max(stock_tree[j, steps] - K, 0.0)
        else:
            opt[j, steps] = max(K - stock_tree[j, steps], 0.0)

    disc = np.exp(-r * dt)

    # Backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold = disc * (p * opt[j, i + 1] + (1.0 - p) * opt[j + 1, i + 1])
            if is_american:
                if is_call:
                    intrinsic = max(stock_tree[j, i] - K, 0.0)
                else:
                    intrinsic = max(K - stock_tree[j, i], 0.0)
                opt[j, i] = max(hold, intrinsic)
            else:
                opt[j, i] = hold

    return opt


# ─────────────────────────────────────────────────────────────────────────────
# High-level pricer class
# ─────────────────────────────────────────────────────────────────────────────

class BinomialOptionPricer:
    """
    CRR binomial option pricer for European and American options.

    Parameters
    ----------
    S0 : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float
        Implied (or assumed) volatility (annualised).
    q : float, optional
        Continuous dividend yield (annualised). Defaults to ``0.0``.
    steps : int, optional
        Number of CRR time steps. Higher values improve accuracy at the
        cost of O(N²) runtime. Defaults to ``50``.
    option_type : str, optional
        ``"call"`` or ``"put"``. Case-insensitive. Defaults to ``"call"``.
    style : str, optional
        ``"american"`` or ``"european"``. Case-insensitive.
        Defaults to ``"american"``.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        steps: int = 50,
        option_type: str = "call",
        style: str = "american",
    ) -> None:
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.steps = int(steps)
        self.option_type = option_type.lower()
        self.style = style.lower()

        self.dt = self.T / self.steps
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u

        raw_p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)
        self.p = float(np.clip(raw_p, 1e-9, 1.0 - 1e-9))

    def price(self) -> Dict[str, object]:
        """
        Compute the option fair value and return the full lattice state.

        Returns
        -------
        dict
            Keys: ``"price"`` (float), ``"stock_tree"`` (ndarray),
            ``"option_tree"`` (ndarray), ``"p"``, ``"u"``, ``"d"``, ``"dt"``.
        """
        stk_tree = fast_build_stock_tree(self.S0, self.u, self.d, self.steps)
        opt_tree = fast_build_option_tree(
            stk_tree,
            self.K,
            self.r,
            self.dt,
            self.p,
            self.steps,
            self.option_type == "call",
            self.style == "american",
        )
        return {
            "price": float(opt_tree[0, 0]),
            "stock_tree": stk_tree,
            "option_tree": opt_tree,
            "p": self.p,
            "u": self.u,
            "d": self.d,
            "dt": self.dt,
        }

    def delta(self) -> float:
        """
        Compute the first-order delta using the first-step finite difference.

        Delta is approximated as::

            delta = (V_up - V_down) / (S_up - S_down)

        where ``V_up`` / ``V_down`` are option values and ``S_up`` /
        ``S_down`` are stock prices at the end of the first time step.

        Returns
        -------
        float
            Option delta (dV/dS). Returns ``0.0`` if the denominator is
            negligibly small.
        """
        out = self.price()
        stk = out["stock_tree"]
        opt = out["option_tree"]

        S_up, S_down = stk[0, 1], stk[1, 1]
        V_up, V_down = opt[0, 1], opt[1, 1]

        denom = S_up - S_down
        if abs(denom) < 1e-12:
            return 0.0
        return float((V_up - V_down) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Functional helpers
# ─────────────────────────────────────────────────────────────────────────────

def crr_price_given_sigma(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    steps: int,
    option_type: str,
    style: str,
) -> float:
    """
    Return the CRR option price for a given volatility level.

    Convenience wrapper around ``BinomialOptionPricer`` for use in
    vectorised implied-volatility solvers and surface enrichment loops.

    Parameters
    ----------
    S0, K, T, r, q, sigma, steps, option_type, style :
        See ``BinomialOptionPricer`` for parameter descriptions.

    Returns
    -------
    float
        Option fair value.
    """
    return BinomialOptionPricer(
        S0=S0, K=K, T=T, r=r, q=q,
        sigma=sigma, steps=steps,
        option_type=option_type, style=style,
    ).price()["price"]


def implied_vol_crr(
    market_price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    steps: int = 80,
    option_type: str = "call",
    style: str = "american",
    vol_low: float = 0.01,
    vol_high: float = 2.50,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """
    Invert the CRR pricing function to recover implied volatility.

    Uses bisection search on the CRR price as a function of sigma.
    The search bracket ``[vol_low, vol_high]`` must straddle the root;
    if it does not (e.g. the option is too deep ITM/OTM for the bracket),
    ``np.nan`` is returned.

    Parameters
    ----------
    market_price : float
        Observed market mid price of the option.
    S0, K, T, r, q :
        Spot, strike, time-to-expiry (years), risk-free rate, dividend yield.
    steps : int
        Number of CRR tree steps used for each pricing evaluation.
    option_type : str
        ``"call"`` or ``"put"``.
    style : str
        ``"american"`` or ``"european"``.
    vol_low : float
        Lower bound of the volatility search bracket. Defaults to ``0.01``.
    vol_high : float
        Upper bound of the volatility search bracket. Defaults to ``2.50``.
    tol : float
        Convergence tolerance on the price residual. Defaults to ``1e-5``.
    max_iter : int
        Maximum bisection iterations. Defaults to ``100``.

    Returns
    -------
    float
        Implied volatility, or ``np.nan`` if the root cannot be bracketed or
        if any input is non-positive / zero.
    """
    if market_price <= 0 or T <= 0 or S0 <= 0 or K <= 0:
        return np.nan

    low, high = vol_low, vol_high

    f_low = crr_price_given_sigma(S0, K, T, r, q, low, steps, option_type, style) - market_price
    f_high = crr_price_given_sigma(S0, K, T, r, q, high, steps, option_type, style) - market_price

    if not (np.isfinite(f_low) and np.isfinite(f_high)):
        return np.nan

    # Root must be bracketed
    if f_low * f_high > 0:
        return np.nan

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = crr_price_given_sigma(S0, K, T, r, q, mid, steps, option_type, style) - market_price

        if abs(f_mid) < tol:
            return mid

        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)
