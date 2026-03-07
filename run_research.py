from __future__ import annotations
import os

from config import GlobalConfig
from universe import BEST_BUCKET, WORST_BUCKET
from strategy import StrategyConfig
from backtest import run_bucket_backtest
from metrics import compare_results


def main():
    cfg = GlobalConfig()

    if cfg.api_key == "YOUR_API_KEY_HERE":
        raise RuntimeError("Please set POLYGON_API_KEY or edit config.py")

    strat_cfg = StrategyConfig(
        entry_price_edge=0.15,       # Targets the top ~12% of price mispricings
        entry_iv_edge=0.003,         # Targets the top ~10% of IV mispricings
        exit_iv_edge=0.001,          # Close the trade when IV edge decays
        max_holding_days=10,
        min_volume=1,
        max_open_positions=5,
        delta_rehedge_threshold_shares=4.0,
        min_chain_size=6,
        max_spread_frac=0.60,
        trade_rich_options=True,
        trade_cheap_options=True,
    )

    print("Running best bucket backtest...")
    best = run_bucket_backtest(
        api_key=cfg.api_key,
        bucket=BEST_BUCKET,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        risk_free_rate=cfg.risk_free_rate,
        dividend_yield=cfg.dividend_yield,
        crr_steps=cfg.crr_steps,
        assumed_spread_bps=cfg.assumed_spread_bps,
        strategy_config=strat_cfg,
    )

    print("Running weak bucket backtest...")
    worst = run_bucket_backtest(
        api_key=cfg.api_key,
        bucket=WORST_BUCKET,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        risk_free_rate=cfg.risk_free_rate,
        dividend_yield=cfg.dividend_yield,
        crr_steps=cfg.crr_steps,
        assumed_spread_bps=cfg.assumed_spread_bps,
        strategy_config=strat_cfg,
    )

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

    print("\nSaved files:")
    print(" outputs/equity_best_bucket.csv")
    print(" outputs/trades_best_bucket.csv")
    print(" outputs/signals_best_bucket.csv")
    print(" outputs/equity_worst_bucket.csv")
    print(" outputs/trades_worst_bucket.csv")
    print(" outputs/signals_worst_bucket.csv")
    print(" outputs/bucket_strategy_comparison.csv")


if __name__ == "__main__":
    main()
