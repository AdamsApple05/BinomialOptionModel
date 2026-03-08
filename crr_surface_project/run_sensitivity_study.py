import time
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from config import GlobalConfig
from strategy import StrategyConfig
from universe import BEST_BUCKET
from backtest import run_bucket_backtest


def run_simulation(steps, cfg, strat_cfg):
    """Worker function for parallel backtests."""
    try:
        start_time = time.time()
        # Note: Internal progress bars in strategy.py should be disabled for parallel runs
        output = run_bucket_backtest(
            api_key=cfg.api_key,
            bucket=BEST_BUCKET,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            risk_free_rate=cfg.risk_free_rate,
            dividend_yield=cfg.dividend_yield,
            crr_steps=steps,
            assumed_spread_bps=cfg.assumed_spread_bps,
            strategy_config=strat_cfg
        )
        duration = time.time() - start_time
        daily_pnl = output["equity"]["daily_pnl"].fillna(0)
        sharpe = (daily_pnl.mean() / daily_pnl.std() *
                  (252**0.5)) if daily_pnl.std() > 0 else 0

        return {
            "steps": steps,
            "sharpe": round(sharpe, 4),
            "total_pnl": round(daily_pnl.sum(), 2),
            "time_sec": round(duration, 2)
        }
    except Exception as e:
        return {"steps": steps, "error": str(e)}


def main():
    cfg = GlobalConfig()
    strat_cfg = StrategyConfig(
        entry_price_edge=0.15, entry_iv_edge=0.003, max_open_positions=5)
    step_sizes = range(25, 325, 25)  # 25, 50, 75... 300
    results = []

    print(f"Launching Parallel Sensitivity Sweep (25-step increments...")

    # Use max_workers=4 to stay safe with API rate limits
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(
            run_simulation, s, cfg, strat_cfg): s for s in step_sizes}
        for future in tqdm(as_completed(futures), total=len(step_sizes), desc="Total Progress"):
            res = future.result()
            if "error" not in res:
                results.append(res)

    # 1. Save Data Points to CSV
    df = pd.DataFrame(results).sort_values("steps")
    df.to_csv("outputs/sensitivity_analysis_data.csv", index=False)
    print("\n✅ Data points saved to outputs/sensitivity_analysis_data.csv")

    # 2. Save High-Res Curve Photo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(df['steps'], df['sharpe'], marker='o',
             color='#1f77b4', linewidth=2.5)
    ax1.set_title('Sharpe Ratio Convergence', fontweight='bold')
    ax1.set_xlabel('CRR Steps')
    ax1.set_ylabel('Annualized Sharpe')
    ax1.grid(True, alpha=0.3)

    ax2.plot(df['steps'], df['time_sec'], marker='s',
             color='#d62728', linewidth=2.5)
    ax2.set_title('Execution Time Scaling (O(N²))', fontweight='bold')
    ax2.set_xlabel('CRR Steps')
    ax2.set_ylabel('Time (Seconds)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/sensitivity_analysis_curve.png')
    print("✅ Curve photo saved to outputs/sensitivity_analysis_curve.png")


if __name__ == "__main__":
    main()
