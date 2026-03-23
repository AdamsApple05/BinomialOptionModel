"""
run_research.py
Quick-start research backtest using live Polygon API data.

Runs the CRR Surface Delta-Hedge Strategy over the date window defined in
``GlobalConfig`` (defaults to Q1 2024) for both the BEST and WORST buckets
and writes summary CSVs to ``outputs/``.

For a multi-year backtest using locally cached data, use
``run_full_backtest.py`` instead (requires the cache to be populated first
via ``run_download.py``).

Usage
-----
Set the ``POLYGON_API_KEY`` environment variable, then from the project root::

    python crr_surface_project/run_research.py
"""

from __future__ import annotations

import os

from config import GlobalConfig
from universe import BEST_BUCKET, WORST_BUCKET
from strategy import StrategyConfig
from backtest import run_bucket_backtest
from metrics import compare_results


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
        exit_iv_edge=0.0,   # disabled — not part of the validated param sweep
        max_holding_days=10,
        min_volume=1,
        max_open_positions=5,
        delta_rehedge_threshold_shares=4.0,
        min_chain_size=6,
        max_spread_frac=0.60,
        trade_rich_options=True,
        trade_cheap_options=True,
    )

    common = dict(
        api_key=cfg.api_key,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        risk_free_rate=cfg.risk_free_rate,
        dividend_yield=cfg.dividend_yield,
        crr_steps=cfg.crr_steps,
        assumed_spread_bps=cfg.assumed_spread_bps,
        strategy_config=strat_cfg,
    )

    print(f"Running BEST bucket backtest  [{cfg.start_date} -> {cfg.end_date}] ...")
    best = run_bucket_backtest(bucket=BEST_BUCKET, **common)

    print(f"Running WORST bucket backtest [{cfg.start_date} -> {cfg.end_date}] ...")
    worst = run_bucket_backtest(bucket=WORST_BUCKET, **common)

    os.makedirs("outputs", exist_ok=True)

    best["equity"].to_csv("outputs/equity_best_bucket.csv", index=False)
    best["trades"].to_csv("outputs/trades_best_bucket.csv", index=False)
    best["signals"].to_csv("outputs/signals_best_bucket.csv", index=False)

    worst["equity"].to_csv("outputs/equity_worst_bucket.csv", index=False)
    worst["trades"].to_csv("outputs/trades_worst_bucket.csv", index=False)
    worst["signals"].to_csv("outputs/signals_worst_bucket.csv", index=False)

    summary = compare_results({
        BEST_BUCKET.label: best,
        WORST_BUCKET.label: worst,
    })
    summary.to_csv("outputs/bucket_strategy_comparison.csv", index=False)

    print("\nBucket comparison:")
    print(summary.to_string(index=False))

    print("\nOutputs written to outputs/")


if __name__ == "__main__":
    main()
