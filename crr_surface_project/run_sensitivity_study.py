"""
run_sensitivity_study.py
Parallel sensitivity sweep: CRR step count vs. strategy performance.

Runs the BEST_BUCKET backtest for each step count in the configured range
using a ``ProcessPoolExecutor`` and produces:

* ``outputs/sensitivity_analysis_data.csv`` — tabular results.
* ``outputs/sensitivity_analysis_curve.png`` — dual-panel plot of Sharpe
  ratio and execution time vs. step count.

The parallel sweep uses ``max_workers=4`` to limit simultaneous live API
connections. Adjust downward on free-tier API keys to avoid rate-limit errors.

Usage
-----
Set ``POLYGON_API_KEY``, then from the project root::

    python crr_surface_project/run_sensitivity_study.py

Note
----
For a more rigorous mathematical convergence analysis (price error vs. N,
delta error vs. N, timing with Numba warm-up) see ``run_convergence.py``.
This script measures strategy-level Sharpe sensitivity, which is noisier
but more directly relevant to production parameter selection.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from config import GlobalConfig
from strategy import StrategyConfig
from universe import BEST_BUCKET
from backtest import run_bucket_backtest


def _run_one(steps: int, cfg: GlobalConfig, strat_cfg: StrategyConfig) -> dict:
    """
    Worker function: run a single backtest at ``steps`` CRR steps.

    Designed to be called from a ``ProcessPoolExecutor`` worker process.

    Parameters
    ----------
    steps : int
        CRR binomial tree step count for this trial.
    cfg : GlobalConfig
        Global configuration (API key, dates, pricing parameters).
    strat_cfg : StrategyConfig
        Strategy hyperparameters (shared across all trials).

    Returns
    -------
    dict
        Keys: ``steps``, ``sharpe``, ``total_pnl``, ``time_sec``.
        On failure: ``steps`` and ``error``.
    """
    try:
        t0 = time.time()
        result = run_bucket_backtest(
            api_key=cfg.api_key,
            bucket=BEST_BUCKET,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            risk_free_rate=cfg.risk_free_rate,
            dividend_yield=cfg.dividend_yield,
            crr_steps=steps,
            assumed_spread_bps=cfg.assumed_spread_bps,
            strategy_config=strat_cfg,
        )
        elapsed = time.time() - t0

        daily_pnl = result["equity"]["daily_pnl"].fillna(0.0)
        std = daily_pnl.std()
        sharpe = float((daily_pnl.mean() / std) * (252 ** 0.5)) if std > 0 else 0.0

        return {
            "steps": steps,
            "sharpe": round(sharpe, 4),
            "total_pnl": round(float(daily_pnl.sum()), 2),
            "time_sec": round(elapsed, 2),
        }
    except Exception as exc:
        return {"steps": steps, "error": str(exc)}


def main() -> None:
    cfg = GlobalConfig()

    if cfg.api_key == "YOUR_API_KEY_HERE":
        raise RuntimeError(
            "Polygon API key not set. "
            "Export POLYGON_API_KEY or edit config.py before running."
        )

    strat_cfg = StrategyConfig(
        entry_price_edge=0.15,
        entry_iv_edge=0.003,
        max_open_positions=5,
    )

    step_sizes = list(range(25, 325, 25))  # 25, 50, 75, …, 300

    print(f"Sensitivity sweep: {len(step_sizes)} trials, "
          f"step range {step_sizes[0]}-{step_sizes[-1]}")
    print(f"Backtest window: {cfg.start_date} -> {cfg.end_date}")
    print(f"Bucket: {BEST_BUCKET.label}")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_one, s, cfg, strat_cfg): s
            for s in step_sizes
        }
        for future in tqdm(
            as_completed(futures), total=len(step_sizes), desc="Sensitivity sweep"
        ):
            res = future.result()
            if "error" not in res:
                results.append(res)
            else:
                print(f"  Warning: steps={res['steps']} failed - {res['error']}")

    if not results:
        print("No successful trials. Check API key and network connection.")
        return

    df = pd.DataFrame(results).sort_values("steps").reset_index(drop=True)
    df.to_csv("outputs/sensitivity_analysis_data.csv", index=False)
    print(f"\nResults saved to outputs/sensitivity_analysis_data.csv")
    print(df.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CRR Step-Count Sensitivity: BEST_BUCKET", fontweight="bold", fontsize=13)

    ax1.plot(df["steps"], df["sharpe"], marker="o", color="#1f77b4", linewidth=2)
    ax1.set_title("Sharpe Ratio vs. CRR Steps")
    ax1.set_xlabel("CRR Steps (N)")
    ax1.set_ylabel("Annualised Sharpe Ratio")
    ax1.grid(True, alpha=0.35)

    ax2.plot(df["steps"], df["time_sec"], marker="s", color="#d62728", linewidth=2)
    ax2.set_title("Execution Time vs. CRR Steps  [O(N²)]")
    ax2.set_xlabel("CRR Steps (N)")
    ax2.set_ylabel("Total Backtest Time (seconds)")
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig("outputs/sensitivity_analysis_curve.png", dpi=150, bbox_inches="tight")
    print("Plot saved to outputs/sensitivity_analysis_curve.png")


if __name__ == "__main__":
    main()
