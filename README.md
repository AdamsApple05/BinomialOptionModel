# Delta-Neutral Volatility Arbitrage Engine

A high-performance quantitative research framework designed to backtest options mispricing strategies. This engine utilizes a dynamically delta-hedged Z-Score approach, evaluating market premiums against a Numba-optimized Binomial Option Pricing model.

## Core Methodology

The strategy identifies volatility arbitrage opportunities by monitoring the statistical deviation (Z-Score) of market prices from a theoretical baseline. To maintain a market-neutral profile, the engine dynamically rebalances a stock hedge based on real-time Delta calculations.

### Key Features

* **High-Speed Pricing Engine**: Custom Binomial Option Pricer optimized with `@njit` (Numba) for rapid calculation of theoretical prices and Greeks (Delta, Gamma, Theta).
* **Hyperparameter Optimization**: Integrated random search module to identify robust parameter clusters and maximize risk-adjusted returns (Sharpe Ratio).
* **Adaptive Volatility Baselines**: Implementation of Exponentially Weighted Moving Averages (EWMA) for reactive historical volatility modeling.
* **Skew-Adjusted Signal Logic**: Incorporates a Put-Call skew offset to account for structural market fear premiums, preventing false positive signals on Put contracts.
* **Dynamic Gamma Management**: Delta hedge thresholds scale automatically based on Time-to-Maturity (TTM), reducing transaction costs and mitigating high-gamma risks near expiration.
* **Institutional Risk Infrastructure**:
    * **Notional Margin Floor**: Hardcoded margin minimums to prevent sizing anomalies during low net-cost setups.
    * **Hard Stop-Loss**: Automatic liquidation if a single trade exceeds a preset Mark-to-Market drawdown limit.
    * **Terminal Liquidation**: Forces closure of active trades at the end of the data horizon to ensure accurate final P&L reporting.

## Project Structure

* `main.py`: The multiprocessing entry point. Houses the universe configuration (`CONFIGS`) and global strategy hyperparameters (`SHARED`).
* `optimizer.py`: Hyperparameter optimization tool utilizing random search to identify optimal parameter neighborhoods across various market regimes.
* `backtest_engine.py`: The execution loop. Iterates over historical data, manages capital, prices options via the binomial model, and compiles portfolio metrics.
* `zscore_strategy.py`: The core logic class. Handles the evaluation of mispricing, dynamic margin-based position sizing, and delta rebalancing.
* `market_data.py`: A thin wrapper around the `polygon-api-client` to fetch historical OHLCV data and daily option aggregates.
* `binomial_pricer.py`: Mathematical model evaluating theoretical option premiums and Greeks.

## Installation

This project requires **Python 3.9+**.

```bash
pip install numpy pandas matplotlib numba polygon-api-client