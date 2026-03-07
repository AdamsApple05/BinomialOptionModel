from __future__ import annotations
import numpy as np
from numba import njit
from typing import Dict


@njit(cache=True)
def fast_build_stock_tree(S0: float, u: float, d: float, steps: int) -> np.ndarray:
    tree = np.zeros((steps + 1, steps + 1))
    tree[0, 0] = S0
    for i in range(1, steps + 1):
        tree[0, i] = tree[0, i - 1] * u
        for j in range(1, i + 1):
            tree[j, i] = tree[j - 1, i - 1] * d
    return tree


@njit(cache=True)
def fast_build_option_tree(stock_tree, K, r, dt, p, steps, is_call, is_american):
    opt = np.zeros_like(stock_tree)

    for j in range(steps + 1):
        if is_call:
            opt[j, steps] = max(stock_tree[j, steps] - K, 0.0)
        else:
            opt[j, steps] = max(K - stock_tree[j, steps], 0.0)

    disc = np.exp(-r * dt)

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold = disc * (p * opt[j, i + 1] + (1.0 - p) * opt[j + 1, i + 1])
            if is_american:
                intrinsic = max(stock_tree[j, i] - K, 0.0) if is_call else max(K - stock_tree[j, i], 0.0)
                opt[j, i] = max(hold, intrinsic)
            else:
                opt[j, i] = hold

    return opt


class BinomialOptionPricer:
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
    ):
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
        self.p = np.clip(raw_p, 1e-9, 1.0 - 1e-9)

    def price(self) -> Dict:
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
            "p": float(self.p),
            "u": float(self.u),
            "d": float(self.d),
            "dt": float(self.dt),
        }

    def delta(self) -> float:
        out = self.price()
        stk = out["stock_tree"]
        opt = out["option_tree"]

        S_up = stk[0, 1]
        S_down = stk[1, 1]
        V_up = opt[0, 1]
        V_down = opt[1, 1]

        denom = S_up - S_down
        if abs(denom) < 1e-12:
            return 0.0
        return float((V_up - V_down) / denom)


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
    return BinomialOptionPricer(
        S0=S0,
        K=K,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        steps=steps,
        option_type=option_type,
        style=style,
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
    if market_price <= 0 or T <= 0 or S0 <= 0 or K <= 0:
        return np.nan

    low = vol_low
    high = vol_high

    f_low = crr_price_given_sigma(S0, K, T, r, q, low, steps, option_type, style) - market_price
    f_high = crr_price_given_sigma(S0, K, T, r, q, high, steps, option_type, style) - market_price

    if np.isnan(f_low) or np.isnan(f_high):
        return np.nan

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
