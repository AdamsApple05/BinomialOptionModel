import numpy as np
from numba import njit
from typing import Dict


@njit(cache=True)
def fast_build_stock_tree(S0: float, u: float, d: float, steps: int) -> np.ndarray:
    """JIT-compiled binomial stock price tree construction using CRR methodology."""
    tree = np.zeros((steps + 1, steps + 1))
    tree[0, 0] = S0
    for i in range(1, steps + 1):
        tree[0, i] = tree[0, i - 1] * u
        for j in range(1, i + 1):
            tree[j, i] = tree[j - 1, i - 1] * d
    return tree


@njit(cache=True)
def fast_build_option_tree(stock_tree, K, r, dt, p, steps, is_call, is_american):
    """JIT-compiled backward induction. Supports early exercise for American styles."""
    opt = np.zeros_like(stock_tree)
    for j in range(steps + 1):
        if is_call:
            opt[j, steps] = max(stock_tree[j, steps] - K, 0.0)
        else:
            opt[j, steps] = max(K - stock_tree[j, steps], 0.0)

    disc = np.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold = disc * (p * opt[j, i + 1] + (1 - p) * opt[j + 1, i + 1])
            if is_american:
                intrinsic = max(
                    stock_tree[j, i] - K, 0.0) if is_call else max(K - stock_tree[j, i], 0.0)
                opt[j, i] = max(hold, intrinsic)
            else:
                opt[j, i] = hold
    return opt


class BinomialOptionPricer:
    """Professional wrapper for Numba-accelerated Binomial pricing."""

    def __init__(self, S0, K, T, r, sigma, steps=50, option_type="call", style="american"):
        self.S0, self.K, self.T, self.r, self.sigma, self.steps = float(
            S0), float(K), float(T), float(r), float(sigma), int(steps)
        self.option_type, self.style = option_type.lower(), style.lower()
        self.dt = self.T / self.steps
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = np.clip((np.exp(self.r * self.dt) - self.d) /
                         (self.u - self.d), 1e-9, 1-1e-9)

    def price(self) -> Dict:
        """Returns the calculated fair value and trees."""
        stk_tree = fast_build_stock_tree(self.S0, self.u, self.d, self.steps)
        opt_tree = fast_build_option_tree(stk_tree, self.K, self.r, self.dt, self.p,
                                          self.steps, self.option_type == "call", self.style == "american")
        return {"price": float(opt_tree[0, 0]), "stock_tree": stk_tree, "option_tree": opt_tree}
