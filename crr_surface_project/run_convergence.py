"""
run_convergence.py
Runs the CRR step-count convergence analysis using SPY-representative parameters.

Usage (from project root):
    python crr_surface_project/run_convergence.py

Or from within crr_surface_project/:
    python run_convergence.py

Outputs: outputs/convergence/*.csv and outputs/convergence/*.png
Runtime: ~2-5 minutes (pure CPU, no API calls)
"""
from __future__ import annotations

from pathlib import Path

from config import GlobalConfig
from convergence_analysis import (
    DEFAULT_TEST_CASES,
    run_full_convergence_suite,
    plot_convergence_curves,
    find_convergence_knee,
)

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "convergence"


def main():
    cfg = GlobalConfig()

    print("=" * 60)
    print("CRR Binomial Step-Count Convergence Analysis")
    print("=" * 60)
    print(f"Parameters: r={cfg.risk_free_rate}, q={cfg.dividend_yield}, sigma=0.18")
    print(f"Test cases: {len(DEFAULT_TEST_CASES)}")
    print(f"Price convergence: N = 2, 4, ..., 500 (even steps for symmetry)")
    print(f"Timing analysis:   N = 10, 20, ..., 300 (10 repeats each)")
    print(f"Output directory:  {OUTPUT_DIR}")
    print()
    print("Note: First run triggers Numba JIT compilation (~30s warmup).")
    print()

    results = run_full_convergence_suite(
        r=cfg.risk_free_rate,
        q=cfg.dividend_yield,
        sigma=0.18,
        test_cases=DEFAULT_TEST_CASES,
        output_dir=OUTPUT_DIR,
        price_step_range=list(range(2, 502, 2)),
        timing_step_range=list(range(10, 310, 10)),
    )

    print()
    print("=== Convergence Summary ===")
    print(f"{'Case':<25} {'Knee N':>8} {'Error@N=80':>12} {'Error@N=10':>12}")
    print("-" * 60)
    for label, data in results.items():
        df = data["convergence"]
        knee = find_convergence_knee(df, "abs_error")
        err_80 = df.loc[df["steps"] == 80, "abs_error"].values
        err_10 = df.loc[df["steps"] == 10, "abs_error"].values
        err_80_str = f"${err_80[0]:.4f}" if len(err_80) > 0 else "N/A"
        err_10_str = f"${err_10[0]:.4f}" if len(err_10) > 0 else "N/A"
        print(f"{label:<25} {knee:>8} {err_80_str:>12} {err_10_str:>12}")

    print()
    print("Generating plots ...")
    plot_convergence_curves(results, output_dir=OUTPUT_DIR)

    print()
    print("Done. Files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
