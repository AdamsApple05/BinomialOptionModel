"""
run_full_backtest.py
Multi-year backtest using locally cached Polygon data (2022-2024).

Usage (from project root):
    python crr_surface_project/run_full_backtest.py

Or from within crr_surface_project/:
    python run_full_backtest.py

Requires: run_download.py to have been run first to populate the cache.
Outputs:  outputs/multi_year/*.csv and outputs/multi_year/metrics_summary.json
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import GlobalConfig
from universe import BEST_BUCKET, WORST_BUCKET
from strategy import StrategyConfig
from backtest import run_bucket_backtest
from data_cache import DataCache, CachedMarketData, CachedUniverseBuilder
from extended_metrics import full_performance_summary

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "multi_year"

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"


def _json_safe(obj):
    """Recursively convert non-JSON-serializable types for json.dump."""
    if isinstance(obj, dict):
        return {
            (str(k.date()) if isinstance(k, pd.Timestamp) else k): _json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, pd.Series):
        return {str(k.date()): _json_safe(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def main():
    cfg = GlobalConfig()

    if cfg.api_key == "YOUR_API_KEY_HERE":
        raise RuntimeError("Please set POLYGON_API_KEY environment variable.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    strat_cfg = StrategyConfig(
        entry_price_edge=0.10,
        entry_iv_edge=0.002,
        exit_iv_edge=0.001,
        max_holding_days=10,
        min_volume=1,
        max_open_positions=10,
        delta_rehedge_threshold_shares=4.0,
        min_chain_size=6,
        max_spread_frac=0.60,
        trade_rich_options=True,
        trade_cheap_options=True,
        dd_min_positions=2,
        max_contracts=3,
        use_signal_strength_sizing=True,
    )

    cache = DataCache(DATA_DIR)
    cached_md = CachedMarketData(api_key=cfg.api_key, cache=cache)
    cached_universe = CachedUniverseBuilder(
        api_key=cfg.api_key, cache=cache, underlying_symbol=cfg.underlying_symbol
    )

    buckets = {
        BEST_BUCKET.label: BEST_BUCKET,
        WORST_BUCKET.label: WORST_BUCKET,
    }

    all_results = {}

    for label, bucket in buckets.items():
        print(f"\n{'=' * 60}")
        print(f"Running backtest: {label}  ({START_DATE} -> {END_DATE})")
        print("=" * 60)

        result = run_bucket_backtest(
            api_key=cfg.api_key,
            bucket=bucket,
            start_date=START_DATE,
            end_date=END_DATE,
            risk_free_rate=cfg.risk_free_rate,
            dividend_yield=cfg.dividend_yield,
            crr_steps=cfg.crr_steps,
            assumed_spread_bps=cfg.assumed_spread_bps,
            strategy_config=strat_cfg,
            market_data_provider=cached_md,
            universe_builder=cached_universe,
        )
        all_results[label] = result

        # Save CSVs
        result["equity"].to_csv(
            OUTPUT_DIR / f"equity_{label}.csv", index=False)
        result["trades"].to_csv(
            OUTPUT_DIR / f"trades_{label}.csv", index=False)
        result["signals"].to_csv(
            OUTPUT_DIR / f"signals_{label}.csv", index=False)

        eq = result["equity"]
        pnl = eq["daily_pnl"].fillna(0.0)
        print(f"  Days: {len(eq)}")
        print(f"  Total P&L: ${pnl.sum():,.2f}")
        print(
            f"  Entries: {(result['trades']['action'].str.startswith('ENTRY')).sum() if not result['trades'].empty else 0}")

    # Load SPY prices from cache for benchmark comparison
    print("\nLoading SPY prices for benchmark metrics ...")
    spy_frames = []
    for year in [2022, 2023, 2024]:
        if cache.has_underlying("SPY", year):
            spy_frames.append(cache.read_underlying("SPY", year))
    spy_prices = pd.concat(spy_frames) if spy_frames else pd.DataFrame()
    if not spy_prices.empty:
        spy_prices = spy_prices[~spy_prices.index.duplicated(
            keep="first")].sort_index()

    # Compute extended metrics for each bucket
    print("\nComputing extended metrics ...")
    metrics_all = {}
    for label, result in all_results.items():
        if result["equity"].empty:
            continue
        metrics = full_performance_summary(
            equity=result["equity"],
            trades=result["trades"],
            spy_prices=spy_prices,
            risk_free_rate=cfg.risk_free_rate,
        )
        metrics_all[label] = metrics

        # Save per-trade detail
        if isinstance(metrics.get("trade_detail"), pd.DataFrame) and not metrics["trade_detail"].empty:
            metrics["trade_detail"].to_csv(
                OUTPUT_DIR / f"trade_detail_{label}.csv", index=False)
        if isinstance(metrics.get("monthly"), pd.DataFrame) and not metrics["monthly"].empty:
            metrics["monthly"].to_csv(OUTPUT_DIR / f"monthly_pnl_{label}.csv")
        if isinstance(metrics.get("yearly"), pd.DataFrame) and not metrics["yearly"].empty:
            metrics["yearly"].to_csv(
                OUTPUT_DIR / f"yearly_summary_{label}.csv", index=False)

        print(f"\n  {label}:")
        ov = metrics.get("overview", {})
        ra = metrics.get("risk_adjusted", {})
        bm = metrics.get("benchmark", {})
        print(f"    Total P&L:  ${ov.get('total_pnl', 0):>12,.2f}")
        print(f"    Sharpe:     {ra.get('sharpe', float('nan')):>8.3f}")
        print(f"    Sortino:    {ra.get('sortino', float('nan')):>8.3f}")
        print(f"    Calmar:     {ra.get('calmar', float('nan')):>8.3f}")
        print(
            f"    Max DD:     ${metrics['drawdown'].get('max_drawdown', 0):>11,.2f}")
        print(
            f"    IR vs SPY:  {bm.get('information_ratio', float('nan')):>8.3f}")

    # Save metrics summary as JSON (non-DataFrame portions)
    print("\nSaving metrics summary ...")
    metrics_serializable = {}
    for label, metrics in metrics_all.items():
        metrics_serializable[label] = {
            k: _json_safe(v) for k, v in metrics.items()
            if not isinstance(v, (pd.DataFrame, pd.Series))
        }
    with open(OUTPUT_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics_serializable, f, indent=2, default=str)

    # Save SPY prices slice for reporting
    if not spy_prices.empty:
        mask = (spy_prices.index >= pd.Timestamp(START_DATE)) & \
               (spy_prices.index <= pd.Timestamp(END_DATE))
        spy_prices.loc[mask].reset_index().to_csv(
            OUTPUT_DIR / "spy_prices.csv", index=False
        )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nFiles written:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
