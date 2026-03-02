import numpy as np
from numba import njit
from typing import Dict

# ===========================================================================
# Compiled Numba Kernels (High Performance)
# ===========================================================================


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
def fast_build_option_tree(
    stock_tree: np.ndarray,
    K: float,
    r: float,
    dt: float,
    p: float,
    steps: int,
    is_call: bool,
    is_american: bool
) -> np.ndarray:
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
            hold = disc * (p * opt[j, i + 1] + (1 - p) * opt[j + 1, i + 1])
            if is_american:
                if is_call:
                    intrinsic = max(stock_tree[j, i] - K, 0.0)
                else:
                    intrinsic = max(K - stock_tree[j, i], 0.0)
                opt[j, i] = max(hold, intrinsic)
            else:
                opt[j, i] = hold

    return opt


@njit(cache=True)
def fast_calculate_greeks(
    stock_tree: np.ndarray,
    option_tree: np.ndarray,
    dt: float,
    steps: int
):
    delta = np.zeros_like(stock_tree)
    gamma = np.zeros_like(stock_tree)
    theta = np.zeros_like(stock_tree)

    for i in range(steps):
        for j in range(i + 1):
            S_u = stock_tree[j, i + 1]
            S_d = stock_tree[j + 1, i + 1]
            V_u = option_tree[j, i + 1]
            V_d = option_tree[j + 1, i + 1]
            dS = S_u - S_d

            if dS != 0:
                delta[j, i] = (V_u - V_d) / dS

            # Gamma requires two steps ahead
            if i < steps - 1 and j < i:
                du = delta[j, i + 1]
                dd = delta[j + 1, i + 1]
                if dS != 0:
                    gamma[j, i] = (du - dd) / dS

            # Theta
            if i < steps - 1:
                V_now = option_tree[j, i]
                V_next = (option_tree[j, i + 1] +
                          option_tree[j + 1, i + 1]) / 2.0
                theta[j, i] = (V_next - V_now) / dt

    return delta, gamma, theta


# ===========================================================================
# Object-Oriented Wrapper
# ===========================================================================

class BinomialOptionPricer:
    """
    CRR Binomial Option Pricer supporting European and American style.
    Returns price, delta, gamma, theta, and full trees.

    Backend calculations are JIT-compiled with Numba for high performance.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        steps: int = 50,
        option_type: str = "call",
        style: str = "american",
    ):
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.steps = int(steps)
        self.option_type = option_type.lower()
        self.style = style.lower()

        self.dt = T / steps
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)

        # Clamp risk-neutral probability to valid range
        self.p = float(np.clip(self.p, 1e-9, 1 - 1e-9))

        self._stock_tree: np.ndarray | None = None
        self._option_tree: np.ndarray | None = None
        self._delta_tree: np.ndarray | None = None
        self._gamma_tree: np.ndarray | None = None
        self._theta_tree: np.ndarray | None = None

    # ------------------------------------------------------------------
    def price(self, recalculate: bool = False) -> Dict:
        """
        Returns dict with keys:
            price, delta, gamma, theta,
            stock_tree, option_tree, delta_tree, gamma_tree, theta_tree
        """
        if recalculate or self._stock_tree is None:
            # Convert strings to booleans for Numba compatibility
            is_call = self.option_type == "call"
            is_american = self.style == "american"

            # Execute compiled kernels
            self._stock_tree = fast_build_stock_tree(
                self.S0, self.u, self.d, self.steps
            )

            self._option_tree = fast_build_option_tree(
                self._stock_tree, self.K, self.r, self.dt, self.p,
                self.steps, is_call, is_american
            )

            delta, gamma, theta = fast_calculate_greeks(
                self._stock_tree, self._option_tree, self.dt, self.steps
            )

            self._delta_tree = delta
            self._gamma_tree = gamma
            self._theta_tree = theta

        return {
            "price": float(self._option_tree[0, 0]),
            "delta": float(self._delta_tree[0, 0]),
            "gamma": float(self._gamma_tree[0, 0]),
            "theta": float(self._theta_tree[0, 0]),
            "stock_tree": self._stock_tree,
            "option_tree": self._option_tree,
            "delta_tree": self._delta_tree,
            "gamma_tree": self._gamma_tree,
            "theta_tree": self._theta_tree,
        }
