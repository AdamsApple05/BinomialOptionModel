"""
wfa_aggregator.py
-----------------
Aggregates individual Walk-Forward OOS windows into a single continuous curve.
Calculates the 'True' Sharpe Ratio of a dynamically adapting strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def aggregate_wfa_results(wfa_results: list, initial_capital: float = 1_000_000.0):
    """
    Stitches together the OOS metrics from the WalkForwardEngine.
    """
    if not wfa_results:
        print("No results found to aggregate.")
        return

    # 1. Compile the Continuous Equity Curve
    # Note: In a live environment, you'd pull the actual daily equity lists.
    # Here, we aggregate the final P&L of each window to see the total path.
    data = []
    current_equity = initial_capital

    for run in wfa_results:
        pnl = run['metrics'].get('total_pnl', 0.0)
        current_equity += pnl
        data.append({
            "Period": run['test_period'],
            "Lookback": run['params']['lookback'],
            "Z-Sell": run['params']['z_entry_sell'],
            "PnL": pnl,
            "Total_Equity": current_equity
        })

    df = pd.DataFrame(data)

    # 2. Performance Attribution
    total_return = (current_equity - initial_capital) / initial_capital

    # Estimate Sharpe based on window-to-window consistency
    window_rets = df['PnL'] / initial_capital
    sharpe = (window_rets.mean() / window_rets.std()) * \
        np.sqrt(12) if len(window_rets) > 1 else 0

    print("\n" + "="*60)
    print("📈 FINAL WALK-FORWARD AGGREGATED REPORT")
    print("="*60)
    print(f"Total Portfolio Return: {total_return:.2%}")
    print(f"Rolling Sharpe Ratio:   {sharpe:.2f}")
    print(f"Final Account Value:    ${current_equity:,.2f}")
    print("="*60)

    # 3. Visualization
    plt.figure(figsize=(12, 6))
    plt.step(df['Period'], df['Total_Equity'], where='post',
             label='WFA Equity Curve', color='#00ffcc')
    plt.xticks(rotation=45)
    plt.title("Continuous Walk-Forward Equity Curve (Out-of-Sample)")
    plt.ylabel("Account Value ($)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df
