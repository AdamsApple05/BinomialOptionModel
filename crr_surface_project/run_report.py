"""
run_report.py
Generates the institutional PDF report from pre-computed backtest outputs.

Usage (from project root):
    python crr_surface_project/run_report.py

Or from within crr_surface_project/:
    python run_report.py

Prerequisites:
    1. run_download.py  — populates local data cache
    2. run_full_backtest.py — generates outputs/multi_year/*.csv
    3. run_convergence.py   — generates outputs/convergence/*.csv (optional but recommended)

Output: outputs/reports/crr_surface_report.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import GlobalConfig
from universe import BEST_BUCKET, WORST_BUCKET
from extended_metrics import full_performance_summary
from reporting import ReportBuilder

SCRIPT_DIR = Path(__file__).parent
MULTI_YEAR_DIR = SCRIPT_DIR.parent / "outputs" / "multi_year"
CONVERGENCE_DIR = SCRIPT_DIR.parent / "outputs" / "convergence"
REPORT_DIR = SCRIPT_DIR.parent / "outputs" / "reports"

BUCKETS = [BEST_BUCKET, WORST_BUCKET]


def load_backtest_results() -> dict[str, dict]:
    """Load equity, trades, signals CSVs from outputs/multi_year/."""
    results = {}
    for bucket in BUCKETS:
        label = bucket.label
        eq_path = MULTI_YEAR_DIR / f"equity_{label}.csv"
        tr_path = MULTI_YEAR_DIR / f"trades_{label}.csv"
        si_path = MULTI_YEAR_DIR / f"signals_{label}.csv"

        if not eq_path.exists():
            print(f"  Warning: {eq_path} not found. Run run_full_backtest.py first.")
            continue

        equity = pd.read_csv(eq_path, parse_dates=["date"])
        trades = pd.read_csv(tr_path, parse_dates=["date"]) if tr_path.exists() else pd.DataFrame()
        signals = pd.read_csv(si_path, parse_dates=["date"]) if si_path.exists() else pd.DataFrame()

        if not equity.empty and "cum_pnl" not in equity.columns:
            equity["cum_pnl"] = equity["daily_pnl"].cumsum()

        results[label] = {"equity": equity, "trades": trades, "signals": signals}

    return results


def load_spy_prices() -> pd.DataFrame:
    """Load SPY prices from outputs/multi_year/spy_prices.csv or data cache."""
    spy_path = MULTI_YEAR_DIR / "spy_prices.csv"
    if spy_path.exists():
        df = pd.read_csv(spy_path, parse_dates=["date"], index_col="date")
        df.index = pd.to_datetime(df.index)
        return df

    # Try loading from data cache directly
    data_dir = SCRIPT_DIR / "data" / "underlying"
    frames = []
    for year in [2022, 2023, 2024]:
        p = data_dir / f"SPY_{year}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"], index_col="date")
            df.index = pd.to_datetime(df.index)
            frames.append(df)

    if frames:
        combined = pd.concat(frames)
        return combined[~combined.index.duplicated(keep="first")].sort_index()

    print("  Warning: SPY price data not found. Benchmark metrics will be unavailable.")
    return pd.DataFrame()


def load_convergence_results() -> dict | None:
    """Load convergence CSVs from outputs/convergence/."""
    if not CONVERGENCE_DIR.exists():
        print("  Note: No convergence data found. Run run_convergence.py to include in report.")
        return None

    csv_files = list(CONVERGENCE_DIR.glob("convergence_*.csv"))
    timing_files = list(CONVERGENCE_DIR.glob("timing_*.csv"))

    if not csv_files:
        return None

    results = {}
    for conv_path in sorted(csv_files):
        # Extract label from filename: convergence_ATM_Put_30_DTE.csv → ATM Put 30 DTE
        stem = conv_path.stem.replace("convergence_", "").replace("_", " ")
        timing_stem = "timing_" + conv_path.stem.replace("convergence_", "")
        timing_path = CONVERGENCE_DIR / f"{timing_stem}.csv"

        conv_df = pd.read_csv(conv_path)
        timing_df = pd.read_csv(timing_path) if timing_path.exists() else pd.DataFrame()

        if not timing_df.empty:
            results[stem] = {"convergence": conv_df, "timing": timing_df}

    return results if results else None


def main():
    cfg = GlobalConfig()

    print("=" * 60)
    print("CRR Surface Strategy - Institutional Report Generator")
    print("=" * 60)

    # Load data
    print("\nLoading backtest results ...")
    backtest_results = load_backtest_results()
    if not backtest_results:
        raise RuntimeError(
            f"No backtest results found in {MULTI_YEAR_DIR}. "
            "Please run run_full_backtest.py first."
        )
    print(f"  Loaded {len(backtest_results)} bucket(s): {list(backtest_results.keys())}")

    print("\nLoading SPY prices ...")
    spy_prices = load_spy_prices()
    print(f"  SPY prices: {len(spy_prices)} bars" if not spy_prices.empty else "  No SPY data.")

    print("\nLoading convergence results ...")
    convergence_results = load_convergence_results()
    if convergence_results:
        print(f"  Convergence data: {len(convergence_results)} test cases")
    else:
        print("  No convergence data found.")

    # Compute extended metrics
    print("\nComputing extended performance metrics ...")
    metrics = {}
    equity_dict = {}
    for label, result in backtest_results.items():
        equity = result["equity"]
        trades = result["trades"]

        if equity.empty:
            print(f"  Skipping {label}: empty equity data")
            continue

        equity_dict[label] = equity
        metrics[label] = full_performance_summary(
            equity=equity,
            trades=trades,
            spy_prices=spy_prices,
            risk_free_rate=cfg.risk_free_rate,
        )

        ov = metrics[label]["overview"]
        ra = metrics[label]["risk_adjusted"]
        print(f"  {label}: P&L=${ov['total_pnl']:,.0f}, Sharpe={ra['sharpe']:.2f}, "
              f"Sortino={ra['sortino']:.2f}")

    if not metrics:
        raise RuntimeError("No valid metrics computed. Check backtest output files.")

    # Build and save report
    print("\nBuilding PDF report ...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "crr_surface_report.pdf"

    figures_dir = REPORT_DIR / "figures"

    builder = ReportBuilder()
    builder.build(
        equity_dict=equity_dict,
        spy_prices=spy_prices,
        metrics=metrics,
        convergence_results=convergence_results,
    )

    print("\nExporting individual figures as PNG ...")
    builder.save_figures_png(figures_dir)

    builder.save_pdf(report_path)

    print(f"\nReport saved to: {report_path}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
