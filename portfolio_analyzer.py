"""
portfolio_analyzer.py
---------------------
Analyzes the consolidated portfolio CSV and generates a Correlation Matrix.
Corrected for index_col and modern Pandas .ffill().
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Ensure you have this: pip install seaborn


def analyze_master_portfolio(file_path="backtest_results/master_portfolio_equity.csv"):
    try:
        # Corrected 'index_col' and using the modern '.ffill()' method
        portfolio = pd.read_csv(file_path, index_col=0,
                                parse_dates=True).ffill()
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Run main.py first.")
        return

    # 1. Calculate Individual Returns for Correlation
    # We use percentage returns to see how strategies move relative to each other
    returns_df = portfolio.pct_change().dropna()
    corr_matrix = returns_df.corr()

    # 2. Portfolio Aggregation
    # Calculate the global portfolio equity by summing all columns
    portfolio['Global_Equity'] = portfolio.sum(axis=1)
    daily_rets = portfolio['Global_Equity'].pct_change().dropna()

    # Annualized Sharpe (assuming 252 trading days)
    sharpe = (daily_rets.mean() / daily_rets.std()) * \
        np.sqrt(252) if daily_rets.std() != 0 else 0

    # Drawdown calculation
    peak = portfolio['Global_Equity'].cummax()
    drawdown = (portfolio['Global_Equity'] - peak) / peak

    print("="*40)
    print("📈 AGGREGATED PORTFOLIO ANALYSIS")
    print("="*40)
    print(f"Aggregated Sharpe:       {sharpe:.2f}")
    print(f"Max Portfolio Drawdown:  {drawdown.min():.2%}")
    print("="*40)

    # 3. Visualization Suite
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    # Plot 1: Global Equity Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(portfolio.index,
             portfolio['Global_Equity'], color="cyan", linewidth=2)
    ax1.set_title("Global Strategy Equity Curve")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.3)

    # Plot 2: Portfolio Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(portfolio.index, drawdown *
                     100, 0, color="red", alpha=0.3)
    ax2.set_title("Portfolio Drawdown %")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)

    # Plot 3: Correlation Heatmap
    ax3 = fig.add_subplot(gs[:, 1])
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f", ax=ax3)
    ax3.set_title("Strategy Correlation Matrix")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_master_portfolio()
