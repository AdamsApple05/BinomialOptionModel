# CRR Surface Delta-Hedge Strategy

A quantitative options research framework that identifies mispricings in
SPY options using a **Cox-Ross-Rubinstein (CRR) binomial pricing model**
combined with a **cross-sectional implied volatility surface**. Positions
are entered when the CRR fair value deviates from the market price beyond
a defined edge threshold and hedged delta-neutrally via the underlying ETF.

---

## Overview

### Signal Generation

1. Fetch the daily SPY option chain for a pre-defined moneyness / DTE bucket.
2. Invert market prices to implied volatilities using bisection on the CRR model.
3. Fit a cross-sectional IV surface: `IV = β₀ + β₁·ln(K/S) + β₂·ln(K/S)² + β₃·√T`
4. Re-price each contract at its surface-fitted IV to obtain a CRR fair value.
5. Compute the residual: `residual = CRR_fair − market_mid`.
6. Enter **long** when `residual ≥ +edge` (option cheap) or **short** when `residual ≤ −edge` (option rich).

### Delta Hedging

On entry, the strategy computes the CRR delta and immediately hedges with
the equivalent number of underlying SPY shares. The hedge is rebalanced when
the delta error exceeds a configured threshold (configurable; set to 4 shares in the run scripts).

### Option Buckets

| Bucket | Type | DTE | Moneyness | Notes |
|--------|------|-----|-----------|-------|
| `PUT_30_60_OTM` | Put | 30–60 | 0.92–0.99 | Primary bucket — high signal density, accurate CRR fit |
| `CALL_120_150_ATM` | Call | 120–150 | 0.98–1.02 | Control bucket — lower accuracy, sparse signals |

### Backtest Results (Q1 2024, live API)

| Bucket | Sharpe | Total P&L | Max DD | Win Rate | Entries |
|--------|--------|-----------|--------|----------|---------|
| PUT_30_60_OTM | 2.74 | $6,938 | −$974 | 56.7% | 125 |
| CALL_120_150_ATM | 1.23 | $5,225 | −$2,532 | 52.4% | 82 |

---

## Project Structure

```
crr_surface_project/
├── crr_surface_project/        # Core package
│   ├── config.py               # Global configuration (rates, dates, CRR steps)
│   ├── market_data.py          # Polygon.io live data client
│   ├── pricer.py               # CRR binomial pricer + IV bisection solver
│   ├── surface.py              # Cross-sectional IV surface fitting (OLS)
│   ├── universe.py             # Option universe definitions and bucket specs
│   ├── strategy.py             # Main strategy: signal generation + delta hedge
│   ├── backtest.py             # Single-bucket backtest orchestrator
│   ├── metrics.py              # Core performance metrics
│   ├── extended_metrics.py     # Institutional metrics: Sortino, Calmar, regime analysis
│   ├── data_cache.py           # Local CSV cache + cached data providers
│   ├── convergence_analysis.py # CRR step-count convergence vs. Black-Scholes
│   ├── reporting.py            # 15-page institutional PDF report builder
│   ├── utils.py                # Shared utilities
│   ├── run_research.py         # Quick-start: live API backtest (Q1 2024)
│   ├── run_download.py         # One-time data download to local cache
│   ├── run_full_backtest.py    # Multi-year backtest using local cache
│   ├── run_convergence.py      # CRR convergence analysis entry point
│   ├── run_sensitivity_study.py# Parallel Sharpe sensitivity sweep
│   └── run_report.py           # Institutional PDF report generator
├── outputs/                    # Generated CSVs, PNGs, and PDFs (git-ignored)
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A [Polygon.io](https://polygon.io) API key (free tier supported for small
  date ranges; paid tier recommended for multi-year downloads)

### Installation

```bash
git clone https://github.com/your-username/crr_surface_project.git
cd crr_surface_project
pip install -r requirements.txt
export POLYGON_API_KEY="your_key_here"
```

### Run the quick-start backtest (live API, Q1 2024)

```bash
python crr_surface_project/run_research.py
# Outputs: outputs/equity_*.csv, outputs/trades_*.csv, outputs/bucket_strategy_comparison.csv
```

### Full workflow (multi-year, cached data)

```bash
# Step 1: Download 3 years of data to local cache (~minutes on paid tier)
python crr_surface_project/run_download.py

# Step 2: CRR mathematical convergence analysis (~2-5 min, CPU only)
python crr_surface_project/run_convergence.py

# Step 3: Run 2022-2024 backtest using cached data
python crr_surface_project/run_full_backtest.py

# Step 4: Generate the 15-page institutional PDF report
python crr_surface_project/run_report.py
# Output: outputs/reports/crr_surface_report.pdf
```

---

## Configuration

All parameters are centralised in `config.py`:

```python
GlobalConfig(
    api_key            = os.getenv("POLYGON_API_KEY"),
    underlying_symbol  = "SPY",
    risk_free_rate     = 0.045,     # 4.5% (annualised)
    dividend_yield     = 0.012,     # 1.2% (SPY approximate)
    crr_steps          = 80,        # CRR tree depth (see run_convergence.py)
    assumed_spread_bps = 25.0,      # Synthetic bid-ask half-spread
    start_date         = "2024-01-01",
    end_date           = "2024-04-01",
    cache_dir          = "data",    # Local data cache root
    calls_per_minute   = 1000,      # Set to 5 for free-tier API
)
```

Strategy hyperparameters are in `strategy.StrategyConfig`.

---

## CRR Model

The pricer implements the standard CRR parameterisation:

| Parameter | Formula |
|-----------|---------|
| Up factor | `u = exp(σ√Δt)` |
| Down factor | `d = 1/u` |
| Risk-neutral prob | `p = (e^((r−q)Δt) − d) / (u − d)` |
| Discount | `e^(−rΔt)` per step |

American exercise is handled by comparing the hold value to the intrinsic
value at every interior node. The pricing kernels are **Numba JIT-compiled**
for O(N²) performance without leaving pure Python.

See `run_convergence.py` for a rigorous analysis of price error and delta
error as a function of N, with comparison to the Black-Scholes analytical
formula.

---

## Convergence Analysis

The `convergence_analysis.py` module benchmarks the CRR pricer against the
analytical Black-Scholes European price across step counts N = 2…500. It
produces four charts:

1. Price error vs. N (log-log) — demonstrates O(1/N) convergence for even step counts
2. Delta error vs. N (log-log)
3. Computation time vs. N (log-log) — confirms O(N²) scaling
4. Accuracy–speed Pareto frontier — justifies the default N=80

---

## Limitations and Assumptions

- **Bid-ask spread is synthetic**: Polygon daily OHLC does not include
  separate bid/ask prices. A fixed 25 bps spread is applied symmetrically
  around the close price.
- **No transaction costs**: Brokerage commissions and exchange fees are not
  modelled.
- **Daily mark-to-market only**: Intraday price moves and early exercise
  events between daily closes are not captured.
- **Constant rates**: Risk-free rate and dividend yield are held constant
  throughout the backtest.
- **American vs. European**: The CRR model prices American options. For the
  convergence analysis, a European CRR price is compared against the
  analytical Black-Scholes formula.

---

## Dependencies

```
numpy
pandas
numba
scipy
tqdm
polygon-api-client
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Disclaimer

This repository is for **research and educational purposes only**. It does
not constitute investment advice or a solicitation to trade. All backtest
results are hypothetical. Past simulated performance is not indicative of
future results.
