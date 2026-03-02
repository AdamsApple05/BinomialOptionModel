import os
import concurrent.futures
import pandas as pd
import time
from backtest_engine import OptionBacktester

# Strategy Config
STRATEGY_CONFIG = {
    "initial_capital": 1_000_000.0,
    "z_entry_sell": 1.4,
    "lookback": 18,
    "capital_allocation": 0.2,
    "expiry_days": 180,
    "start_date": "2023-01-01",
    "end_date": "2024-01-01"
}

# SAFETY ADJUSTMENT: 10 workers is the sweet spot for a Paid Tier
# to avoid internal Polygon rate-limiting on historical data.
MAX_WORKERS = 10


def execute_ticker(symbol, api_key):
    """Isolated worker logic to prevent thread collision."""
    try:
        # We initialize the engine INSIDE the worker to give it a fresh connection
        tester = OptionBacktester(api_key=api_key)
        params = {k: v for k, v in STRATEGY_CONFIG.items() if k not in [
            "start_date", "end_date", "expiry_days"]}

        tester.run_backtest(
            symbol,
            STRATEGY_CONFIG["start_date"],
            STRATEGY_CONFIG["end_date"],
            STRATEGY_CONFIG["expiry_days"],
            "call",
            **params
        )

        if not tester.equity_curve:
            return {"symbol": symbol, "success": False, "error": "Empty Curve"}

        return {"symbol": symbol, "success": True, "equity_curve": tester.equity_curve}
    except Exception as e:
        return {"symbol": symbol, "success": False, "error": str(e)}


def main():
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("CRITICAL: No API Key found.")
        return

    universe_df = pd.read_csv("institutional_universe.csv")
    tickers = universe_df["ticker"].head(1000).tolist()

    print(f"Starting Robust Simulation: {len(tickers)} symbols...")
    successful_runs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(
            execute_ticker, t, api_key): t for t in tickers}

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result["success"]:
                successful_runs.append(result)
                print(f"DONE: {result['symbol']}")
            else:
                # This will tell us EXACTLY why it skipped in the terminal
                print(
                    f"FAIL: {result['symbol']} | Reason: {result.get('error')}")

    if successful_runs:
        print(f"\nConsolidating {len(successful_runs)} results...")
        os.makedirs("backtest_results", exist_ok=True)
        all_curves = []
        for run in successful_runs:
            df = pd.DataFrame(run["equity_curve"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["equity"]].rename(
                columns={"equity": run["symbol"]})
            all_curves.append(df)

        master_df = pd.concat(all_curves, axis=1).ffill()
        master_df.to_csv("backtest_results/master_portfolio_equity.csv")
        print("SUCCESS: master_portfolio_equity.csv generated.")
    else:
        print("ERROR: Zero tickers succeeded. Check the FAIL reasons above.")


if __name__ == "__main__":
    main()
