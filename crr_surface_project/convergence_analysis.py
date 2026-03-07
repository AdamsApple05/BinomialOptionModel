"""
convergence_analysis.py
Mathematical convergence analysis of the CRR binomial pricer:
  - Price error vs. N steps (compared to Black-Scholes European analytical)
  - Delta error vs. N steps
  - Wall-clock time per call vs. N steps
  - Pareto frontier: accuracy vs. speed
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pricer import BinomialOptionPricer, crr_price_given_sigma


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes analytical formulas (European only)
# Used as the N→∞ reference since CRR converges to European BS
# ─────────────────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erfc (no scipy dependency)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def black_scholes_price(
    S: float, K: float, T: float, r: float, q: float,
    sigma: float, option_type: str = "put"
) -> float:
    """Analytical European Black-Scholes price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = max(K - S, 0.0) if option_type == "put" else max(S - K, 0.0)
        return intrinsic

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def black_scholes_delta(
    S: float, K: float, T: float, r: float, q: float,
    sigma: float, option_type: str = "put"
) -> float:
    """Analytical European Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    if option_type == "call":
        return math.exp(-q * T) * _norm_cdf(d1)
    else:
        return -math.exp(-q * T) * _norm_cdf(-d1)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_price_convergence(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: str = "put",
    style: str = "european",
    step_range=None,
) -> pd.DataFrame:
    """
    For each N in step_range, compute CRR price and delta, measure wall-clock
    time, and compute absolute/relative error vs. Black-Scholes European price.

    Returns DataFrame with columns:
        steps, crr_price, bs_price, abs_error, rel_error_pct,
        crr_delta, bs_delta, delta_abs_error, time_us
    """
    if step_range is None:
        step_range = list(range(2, 502, 2))

    bs_price = black_scholes_price(S, K, T, r, q, sigma, option_type)
    bs_delta = black_scholes_delta(S, K, T, r, q, sigma, option_type)

    records = []
    for n in step_range:
        t0 = time.perf_counter()
        pricer = BinomialOptionPricer(
            S0=S, K=K, T=T, r=r, q=q, sigma=sigma,
            steps=n, option_type=option_type, style=style,
        )
        crr_p = pricer.price()["price"]
        crr_d = pricer.delta()
        elapsed_us = (time.perf_counter() - t0) * 1e6

        abs_err = abs(crr_p - bs_price)
        rel_err_pct = (abs_err / bs_price * 100.0) if bs_price > 1e-8 else float("nan")
        delta_err = abs(crr_d - bs_delta)

        records.append({
            "steps": n,
            "crr_price": crr_p,
            "bs_price": bs_price,
            "abs_error": abs_err,
            "rel_error_pct": rel_err_pct,
            "crr_delta": crr_d,
            "bs_delta": bs_delta,
            "delta_abs_error": delta_err,
            "time_us": elapsed_us,
        })

    return pd.DataFrame(records)


def find_convergence_knee(df: pd.DataFrame, error_col: str = "abs_error") -> int:
    """
    Find the 'knee' of the convergence curve using the second-difference method.
    Returns the step count at the knee.
    """
    errors = df[error_col].values
    steps = df["steps"].values

    if len(errors) < 3:
        return int(steps[0])

    # Work on log scale (smoother)
    log_err = np.log(np.maximum(errors, 1e-12))
    second_diff = np.diff(np.diff(log_err))
    # Knee = point of largest curvature (most negative second difference)
    knee_idx = int(np.argmin(second_diff)) + 1
    return int(steps[knee_idx])


def run_timing_analysis(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: str = "put",
    style: str = "european",
    step_range=None,
    repeats: int = 10,
) -> pd.DataFrame:
    """
    Measures median wall-clock time (microseconds) per pricing call.
    Runs a warmup call first to trigger Numba JIT compilation.
    """
    if step_range is None:
        step_range = list(range(10, 510, 10))

    # Numba JIT warmup
    BinomialOptionPricer(S0=S, K=K, T=T, r=r, q=q, sigma=sigma,
                         steps=10, option_type=option_type, style=style).price()

    records = []
    for n in step_range:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            BinomialOptionPricer(
                S0=S, K=K, T=T, r=r, q=q, sigma=sigma,
                steps=n, option_type=option_type, style=style,
            ).price()
            times.append((time.perf_counter() - t0) * 1e6)

        times_arr = np.array(times)
        records.append({
            "steps": n,
            "median_time_us": float(np.median(times_arr)),
            "p25_time_us": float(np.percentile(times_arr, 25)),
            "p75_time_us": float(np.percentile(times_arr, 75)),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Full suite
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TEST_CASES = [
    {"label": "ATM Put 30 DTE",   "S": 480, "K": 480, "T": 30/252,  "option_type": "put"},
    {"label": "ATM Put 60 DTE",   "S": 480, "K": 480, "T": 60/252,  "option_type": "put"},
    {"label": "OTM Put 45 DTE",   "S": 480, "K": 456, "T": 45/252,  "option_type": "put"},
    {"label": "Deep ITM Put",     "S": 480, "K": 530, "T": 30/252,  "option_type": "put"},
    {"label": "ATM Call 135 DTE", "S": 480, "K": 480, "T": 135/252, "option_type": "call"},
]


def run_full_convergence_suite(
    r: float = 0.045,
    q: float = 0.012,
    sigma: float = 0.18,
    test_cases: List[Dict] | None = None,
    output_dir: str | Path = "outputs/convergence",
    price_step_range=None,
    timing_step_range=None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Runs price convergence and timing analysis for a set of representative
    option configurations. Saves CSVs. Returns nested dict of DataFrames.

    Returns: {case_label: {"convergence": df, "timing": df}}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if test_cases is None:
        test_cases = DEFAULT_TEST_CASES
    if price_step_range is None:
        price_step_range = list(range(2, 502, 2))
    if timing_step_range is None:
        timing_step_range = list(range(10, 310, 10))

    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for tc in test_cases:
        label = tc["label"]
        S = tc["S"]
        K = tc["K"]
        T = tc["T"]
        otype = tc["option_type"]
        safe_label = label.replace(" ", "_").replace("/", "-")

        print(f"  Convergence: {label} ...")
        conv_df = run_price_convergence(
            S=S, K=K, T=T, r=r, q=q, sigma=sigma,
            option_type=otype, style="european",
            step_range=price_step_range,
        )
        conv_df.to_csv(output_dir / f"convergence_{safe_label}.csv", index=False)

        print(f"  Timing:      {label} ...")
        timing_df = run_timing_analysis(
            S=S, K=K, T=T, r=r, q=q, sigma=sigma,
            option_type=otype, style="european",
            step_range=timing_step_range,
        )
        timing_df.to_csv(output_dir / f"timing_{safe_label}.csv", index=False)

        results[label] = {"convergence": conv_df, "timing": timing_df}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

RECOMMENDED_N = 80
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
}


def plot_convergence_curves(
    results: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: str | Path = "outputs/convergence",
) -> List[plt.Figure]:
    """
    Produces 4 figures and saves them as PNG:
      1. Price error vs. N (log-log)
      2. Delta error vs. N (log-log)
      3. Time per call vs. N (log-log)
      4. Pareto frontier: error vs. time
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.tab10.colors
    figures = []

    with plt.style.context(STYLE):

        # ── Figure 1: Price Error vs N ────────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        for i, (label, data) in enumerate(results.items()):
            df = data["convergence"]
            ax1.loglog(df["steps"], df["abs_error"], label=label,
                       color=colors[i % len(colors)], linewidth=1.5)
        ax1.axvline(RECOMMENDED_N, color="black", linestyle="--", linewidth=1.2,
                    label=f"N={RECOMMENDED_N} (default)")
        ax1.set_xlabel("Number of CRR Steps (N)", fontsize=12)
        ax1.set_ylabel("Absolute Price Error vs. B-S ($)", fontsize=12)
        ax1.set_title("CRR Binomial Price Convergence to Black-Scholes\n(European exercise, log-log scale)",
                      fontsize=13, fontweight="bold")
        ax1.legend(fontsize=9, loc="upper right")
        ax1.text(0.02, 0.05, "Error ∝ O(1/N) for even steps",
                 transform=ax1.transAxes, fontsize=9, color="gray", style="italic")
        fig1.tight_layout()
        fig1.savefig(output_dir / "fig1_price_error_vs_N.png", dpi=150, bbox_inches="tight")
        figures.append(fig1)

        # ── Figure 2: Delta Error vs N ────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for i, (label, data) in enumerate(results.items()):
            df = data["convergence"]
            ax2.loglog(df["steps"], df["delta_abs_error"], label=label,
                       color=colors[i % len(colors)], linewidth=1.5)
        ax2.axvline(RECOMMENDED_N, color="black", linestyle="--", linewidth=1.2,
                    label=f"N={RECOMMENDED_N} (default)")
        ax2.set_xlabel("Number of CRR Steps (N)", fontsize=12)
        ax2.set_ylabel("Absolute Delta Error vs. B-S", fontsize=12)
        ax2.set_title("CRR Binomial Delta Convergence to Black-Scholes\n(log-log scale)",
                      fontsize=13, fontweight="bold")
        ax2.legend(fontsize=9, loc="upper right")
        fig2.tight_layout()
        fig2.savefig(output_dir / "fig2_delta_error_vs_N.png", dpi=150, bbox_inches="tight")
        figures.append(fig2)

        # ── Figure 3: Time per call vs N ──────────────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        # Use first test case for timing (all cases have same timing profile)
        first_label = list(results.keys())[0]
        timing_df = results[first_label]["timing"]

        ax3.loglog(timing_df["steps"], timing_df["median_time_us"],
                   color=colors[0], linewidth=2, label="Median time")
        ax3.fill_between(
            timing_df["steps"], timing_df["p25_time_us"], timing_df["p75_time_us"],
            alpha=0.2, color=colors[0], label="25th–75th percentile"
        )

        # Fit O(N^2) line
        steps_arr = timing_df["steps"].values.astype(float)
        times_arr = timing_df["median_time_us"].values
        valid = (steps_arr > 50) & np.isfinite(times_arr) & (times_arr > 0)
        if valid.sum() >= 2:
            log_s = np.log(steps_arr[valid])
            log_t = np.log(times_arr[valid])
            slope, intercept = np.polyfit(log_s, log_t, 1)
            fitted = np.exp(intercept) * steps_arr ** slope
            ax3.loglog(steps_arr, fitted, "k--", linewidth=1,
                       label=f"Fitted slope: {slope:.2f} (O(N^{slope:.1f}))")

        ax3.axvline(RECOMMENDED_N, color="black", linestyle=":", linewidth=1.2,
                    label=f"N={RECOMMENDED_N} (default)")
        ax3.set_xlabel("Number of CRR Steps (N)", fontsize=12)
        ax3.set_ylabel("Time per Pricing Call (μs)", fontsize=12)
        ax3.set_title("CRR Binomial Pricer: Computation Time vs. Steps\n(Numba JIT-compiled, log-log scale)",
                      fontsize=13, fontweight="bold")
        ax3.legend(fontsize=9)
        fig3.tight_layout()
        fig3.savefig(output_dir / "fig3_time_vs_N.png", dpi=150, bbox_inches="tight")
        figures.append(fig3)

        # ── Figure 4: Pareto Frontier ─────────────────────────────────────────
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        for i, (label, data) in enumerate(results.items()):
            conv_df = data["convergence"]
            timing_df = results[label]["timing"]

            # Align on step count
            merged = pd.merge(
                conv_df[["steps", "abs_error"]],
                timing_df[["steps", "median_time_us"]],
                on="steps", how="inner"
            )
            sc = ax4.scatter(
                merged["median_time_us"], merged["abs_error"],
                c=[i] * len(merged), cmap="tab10",
                vmin=0, vmax=9, s=20, alpha=0.7, label=label
            )

            # Annotate RECOMMENDED_N
            row = merged[merged["steps"] == RECOMMENDED_N]
            if not row.empty:
                ax4.annotate(
                    f"N={RECOMMENDED_N}",
                    xy=(row["median_time_us"].iloc[0], row["abs_error"].iloc[0]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=8, color="black",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                )

        ax4.set_xlabel("Time per Call (μs)", fontsize=12)
        ax4.set_ylabel("Absolute Price Error vs. B-S ($)", fontsize=12)
        ax4.set_title("Accuracy–Speed Pareto Frontier\n(lower-left is better; each point = one N value)",
                      fontsize=13, fontweight="bold")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.legend(fontsize=9, loc="upper right")
        fig4.tight_layout()
        fig4.savefig(output_dir / "fig4_pareto_frontier.png", dpi=150, bbox_inches="tight")
        figures.append(fig4)

    print(f"Convergence plots saved to {output_dir}")
    return figures
