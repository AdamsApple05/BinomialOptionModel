"""
run_param_sweep.py
Grid-search parameter optimisation on the 2018–2021 train period.

Sweeps entry_price_edge × entry_iv_edge × max_holding_days and selects
the combination that maximises the annualised Sharpe ratio subject to a
maximum-drawdown constraint.  All backtests use the local data cache
(no API calls required).

Usage (from project root):
    python crr_surface_project/run_param_sweep.py

Prerequisites:
    run_download.py must have been run with START_YEAR = 2018 so that
    the 2018-2021 data is present in the local cache.

Outputs (in outputs/param_sweep/):
    sweep_results.csv          All 75 (params, sharpe, max_dd, pnl, win_rate)
    heatmap_sharpe.png         Sharpe heatmap: price_edge x iv_edge (3 subplots for holding_days)
    heatmap_maxdd.png          Max-drawdown heatmap (same layout)
    yearly_breakdown.csv       Per-year Sharpe for the best params (2018-2021)
    best_params.txt            Plain-text record of the chosen params + stats
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import GlobalConfig
from universe import BEST_BUCKET
from strategy import StrategyConfig
from backtest import run_bucket_backtest
from data_cache import DataCache, CachedMarketData, CachedUniverseBuilder
from extended_metrics import sharpe_ratio, drawdown_series

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_START = "2022-01-01"
TRAIN_END   = "2022-12-31"

DATA_DIR   = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "param_sweep"

# Drawdown constraint: reject any param combo whose max drawdown over the
# train period is worse than this value (more negative = bigger loss).
MAX_DD_FLOOR = -8_000.0   # dollars — scaled for 1-year train period

# Grid definition
PRICE_EDGES    = [0.05, 0.10, 0.15, 0.20, 0.30]
IV_EDGES       = [0.001, 0.002, 0.003, 0.005, 0.010]
HOLDING_DAYS   = [5, 10, 15]

# Fixed hyperparameters (not part of the sweep)
FIXED_PARAMS = dict(
    max_open_positions=10,
    delta_rehedge_threshold_shares=4.0,
    max_spread_frac=0.60,
    min_volume=1,
    min_chain_size=6,
    dd_min_positions=2,
    max_contracts=3,
    use_signal_strength_sizing=True,
    trade_rich_options=True,
    trade_cheap_options=True,
    exit_iv_edge=0.001,
)

MAX_WORKERS = 6


# ── Worker (runs in subprocess) ───────────────────────────────────────────────

def _run_one(args: tuple) -> Dict:
    """
    Worker function executed in a separate process.
    Returns a dict of metrics for one (price_edge, iv_edge, holding_days) combo.
    """
    price_edge, iv_edge, holding_days, api_key, data_dir_str = args

    data_dir = Path(data_dir_str)
    cache     = DataCache(data_dir)
    cached_md = CachedMarketData(api_key=api_key, cache=cache)
    cached_uni = CachedUniverseBuilder(
        api_key=api_key, cache=cache, underlying_symbol="SPY"
    )

    strat_cfg = StrategyConfig(
        entry_price_edge=price_edge,
        entry_iv_edge=iv_edge,
        max_holding_days=holding_days,
        **FIXED_PARAMS,
    )

    try:
        t0 = time.monotonic()
        result = run_bucket_backtest(
            api_key=api_key,
            bucket=BEST_BUCKET,
            start_date=TRAIN_START,
            end_date=TRAIN_END,
            risk_free_rate=0.045,
            dividend_yield=0.012,
            crr_steps=80,
            assumed_spread_bps=25.0,
            strategy_config=strat_cfg,
            market_data_provider=cached_md,
            universe_builder=cached_uni,
        )
        elapsed = time.monotonic() - t0

        equity = result["equity"]
        trades = result["trades"]
        pnl    = equity["daily_pnl"].fillna(0.0)
        cum    = pnl.cumsum()
        dd     = float((cum - cum.cummax()).min()) if len(cum) > 0 else 0.0
        sharpe = float(
            (pnl.mean() / pnl.std(ddof=1)) * math.sqrt(252)
        ) if pnl.std(ddof=1) > 0 else 0.0

        entries     = int(trades["action"].str.startswith("ENTRY").sum()) if not trades.empty else 0
        win_rate    = float("nan")
        if not trades.empty:
            entry_rows = trades[trades["action"].str.startswith("ENTRY")]
            exit_rows  = trades[trades["action"] == "EXIT"]
            # rough per-trade win rate based on matched pairs
            matched_pnls = []
            for _, e in entry_rows.iterrows():
                ticker = e["option_ticker"]
                side   = int(e.get("side", 1))
                ep     = float(e.get("market_mid", e.get("option_mid", float("nan"))))
                match  = exit_rows[
                    (exit_rows["option_ticker"] == ticker) &
                    (exit_rows["date"] >= e["date"])
                ].sort_values("date")
                if not match.empty and math.isfinite(ep):
                    xp = float(match.iloc[0].get("option_mid", float("nan")))
                    if math.isfinite(xp):
                        matched_pnls.append(side * 100.0 * (xp - ep))
            if matched_pnls:
                arr = np.array(matched_pnls)
                win_rate = float((arr > 0).mean())

        return {
            "entry_price_edge": price_edge,
            "entry_iv_edge":    iv_edge,
            "max_holding_days": holding_days,
            "sharpe":           round(sharpe, 4),
            "max_drawdown":     round(dd, 2),
            "total_pnl":        round(float(cum.iloc[-1]) if len(cum) > 0 else 0.0, 2),
            "win_rate":         round(win_rate, 4) if math.isfinite(win_rate) else float("nan"),
            "n_entries":        entries,
            "elapsed_sec":      round(elapsed, 1),
            "error":            None,
        }

    except Exception as exc:
        return {
            "entry_price_edge": price_edge,
            "entry_iv_edge":    iv_edge,
            "max_holding_days": holding_days,
            "sharpe":           float("nan"),
            "max_drawdown":     float("nan"),
            "total_pnl":        float("nan"),
            "win_rate":         float("nan"),
            "n_entries":        0,
            "elapsed_sec":      float("nan"),
            "error":            str(exc),
        }


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Produce Sharpe and max-drawdown heatmaps:
    one column per max_holding_days value, rows = iv_edge, cols = price_edge.
    """
    holding_vals = sorted(df["max_holding_days"].unique())
    n_cols = len(holding_vals)

    for metric, title, cmap, fmt in [
        ("sharpe",       "Sharpe Ratio (train 2022)",    "RdYlGn",   ".2f"),
        ("max_drawdown", "Max Drawdown $ (train 2022)",  "RdYlGn_r", ",.0f"),
    ]:
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

        for col_idx, hd in enumerate(holding_vals):
            ax = axes[0][col_idx]
            sub = df[df["max_holding_days"] == hd].copy()

            pivot = sub.pivot_table(
                index="entry_iv_edge",
                columns="entry_price_edge",
                values=metric,
                aggfunc="mean",
            )
            pivot = pivot.sort_index(ascending=False)

            # Mark invalid (DD constraint violated) cells with hatching
            valid_pivot = sub.pivot_table(
                index="entry_iv_edge",
                columns="entry_price_edge",
                values="max_drawdown",
                aggfunc="mean",
            ).sort_index(ascending=False)

            vmin = pivot.values[np.isfinite(pivot.values)].min() if np.any(np.isfinite(pivot.values)) else 0
            vmax = pivot.values[np.isfinite(pivot.values)].max() if np.any(np.isfinite(pivot.values)) else 1

            im = ax.imshow(
                pivot.values,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            # Annotate cells
            for r in range(pivot.shape[0]):
                for c in range(pivot.shape[1]):
                    val = pivot.values[r, c]
                    if not np.isfinite(val):
                        ax.text(c, r, "N/A", ha="center", va="center", fontsize=7, color="gray")
                        continue
                    dd_val = valid_pivot.values[r, c] if r < valid_pivot.shape[0] and c < valid_pivot.shape[1] else 0.0
                    violated = np.isfinite(dd_val) and dd_val < MAX_DD_FLOOR
                    color = "white" if violated else "black"
                    label = (f"({fmt % val})*" if violated else f"{fmt % val}") if "," not in fmt else (
                        f"({'${:,.0f}'.format(val)})*" if violated else f"{'${:,.0f}'.format(val)}"
                    )
                    ax.text(c, r, label, ha="center", va="center", fontsize=7, color=color)

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(v) for v in pivot.columns], fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(v) for v in pivot.index], fontsize=7)
            ax.set_xlabel("entry_price_edge", fontsize=8)
            ax.set_ylabel("entry_iv_edge", fontsize=8)
            ax.set_title(f"max_holding_days = {hd}", fontsize=9)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fname = "heatmap_sharpe.png" if metric == "sharpe" else "heatmap_maxdd.png"
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / fname}")


# ── Per-year breakdown for best params ────────────────────────────────────────

def _yearly_breakdown(
    best_row: pd.Series,
    api_key: str,
    data_dir: Path,
    output_dir: Path,
) -> None:
    """Run one backtest per year (2022-2024) with the best params and save a CSV."""
    years = [2022, 2023, 2024]
    rows  = []

    cache      = DataCache(data_dir)
    cached_md  = CachedMarketData(api_key=api_key, cache=cache)
    cached_uni = CachedUniverseBuilder(api_key=api_key, cache=cache, underlying_symbol="SPY")

    strat_cfg = StrategyConfig(
        entry_price_edge=float(best_row["entry_price_edge"]),
        entry_iv_edge=float(best_row["entry_iv_edge"]),
        max_holding_days=int(best_row["max_holding_days"]),
        **FIXED_PARAMS,
    )

    for yr in years:
        print(f"  Yearly breakdown: {yr} ...")
        result = run_bucket_backtest(
            api_key=api_key,
            bucket=BEST_BUCKET,
            start_date=f"{yr}-01-01",
            end_date=f"{yr}-12-31",
            risk_free_rate=0.045,
            dividend_yield=0.012,
            crr_steps=80,
            assumed_spread_bps=25.0,
            strategy_config=strat_cfg,
            market_data_provider=cached_md,
            universe_builder=cached_uni,
        )

        pnl = result["equity"]["daily_pnl"].fillna(0.0)
        cum = pnl.cumsum()
        dd  = float((cum - cum.cummax()).min()) if len(cum) > 0 else 0.0
        sh  = float((pnl.mean() / pnl.std(ddof=1)) * math.sqrt(252)) if pnl.std(ddof=1) > 0 else float("nan")

        rows.append({
            "year":        yr,
            "total_pnl":   round(float(cum.iloc[-1]) if len(cum) > 0 else 0.0, 2),
            "sharpe":      round(sh, 4),
            "max_drawdown": round(dd, 2),
            "trading_days": len(pnl),
        })

    out = pd.DataFrame(rows)
    path = output_dir / "yearly_breakdown.csv"
    out.to_csv(path, index=False)
    print(f"\n  Per-year breakdown (best params on train set):")
    print(out.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = GlobalConfig()
    if cfg.api_key == "YOUR_API_KEY_HERE":
        raise RuntimeError("Set POLYGON_API_KEY environment variable first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(PRICE_EDGES, IV_EDGES, HOLDING_DAYS))
    total  = len(combos)

    print("=" * 65)
    print("CRR Surface Strategy — Parameter Sweep (Train: 2022)")
    print("=" * 65)
    print(f"Bucket:        {BEST_BUCKET.label}")
    print(f"Combinations:  {total}  (price_edges={len(PRICE_EDGES)}"
          f" × iv_edges={len(IV_EDGES)} × holding={len(HOLDING_DAYS)})")
    print(f"Workers:       {MAX_WORKERS}")
    print(f"DD floor:      ${MAX_DD_FLOOR:,.0f}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print()

    args_list = [
        (pe, ie, hd, cfg.api_key, str(DATA_DIR))
        for pe, ie, hd in combos
    ]

    results: List[Dict] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, a): a for a in args_list}
        for fut in tqdm(as_completed(futures), total=total, desc="Sweep"):
            res = fut.result()
            if res["error"]:
                print(f"\n  [warn] error for "
                      f"price={res['entry_price_edge']} iv={res['entry_iv_edge']} "
                      f"hd={res['max_holding_days']}: {res['error'][:80]}")
            results.append(res)

    df = pd.DataFrame(results).sort_values(
        ["entry_price_edge", "entry_iv_edge", "max_holding_days"]
    ).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep results → {csv_path}")

    # ── Select best params ────────────────────────────────────────────────────
    valid = df[
        (df["max_drawdown"] >= MAX_DD_FLOOR) &
        (df["sharpe"] > 0) &
        df["sharpe"].notna()
    ].copy()

    print(f"\nValid combinations (DD ≥ ${MAX_DD_FLOOR:,.0f} & Sharpe > 0): "
          f"{len(valid)} / {total}")

    if valid.empty:
        print("\n[WARNING] No combinations passed the DD constraint.")
        print("Consider relaxing MAX_DD_FLOOR.  Showing top 10 by Sharpe anyway:")
        valid = df[df["sharpe"].notna()].nlargest(10, "sharpe")

    # Robustness check: flag combinations at the edge of the grid
    # (entry_price_edge == max or entry_iv_edge == max suggest the grid should be wider)
    pe_max = max(PRICE_EDGES)
    ie_max = max(IV_EDGES)
    interior = valid[
        (valid["entry_price_edge"] < pe_max) &
        (valid["entry_iv_edge"] < ie_max)
    ]
    if not interior.empty:
        best = interior.nlargest(1, "sharpe").iloc[0]
        edge_note = ""
    else:
        # Fall back to edge params with a warning
        best = valid.nlargest(1, "sharpe").iloc[0]
        edge_note = " [NOTE: best params are at the edge of the search grid — consider expanding]"

    print(f"\n{'=' * 65}")
    print(f"BEST PARAMS{edge_note}")
    print(f"{'=' * 65}")
    print(f"  entry_price_edge  = {best['entry_price_edge']}")
    print(f"  entry_iv_edge     = {best['entry_iv_edge']}")
    print(f"  max_holding_days  = {int(best['max_holding_days'])}")
    print(f"  Sharpe (train)    = {best['sharpe']:.4f}")
    print(f"  Max DD (train)    = ${best['max_drawdown']:,.2f}")
    print(f"  Total P&L (train) = ${best['total_pnl']:,.2f}")
    print(f"  Win rate          = {best['win_rate']:.2%}" if not math.isnan(best['win_rate']) else "  Win rate = N/A")

    # Save best params to text file
    best_txt = OUTPUT_DIR / "best_params.txt"
    with open(best_txt, "w") as f:
        f.write("Best params selected from train period 2018-2021\n")
        f.write(f"entry_price_edge  = {best['entry_price_edge']}\n")
        f.write(f"entry_iv_edge     = {best['entry_iv_edge']}\n")
        f.write(f"max_holding_days  = {int(best['max_holding_days'])}\n")
        f.write(f"Sharpe (train)    = {best['sharpe']:.4f}\n")
        f.write(f"Max DD (train)    = {best['max_drawdown']:.2f}\n")
        f.write(f"Total P&L (train) = {best['total_pnl']:.2f}\n")
        f.write(f"DD floor used     = {MAX_DD_FLOOR}\n")
        f.write(edge_note + "\n")
    print(f"\nBest params saved → {best_txt}")

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    print("\nGenerating heatmaps ...")
    _plot_heatmaps(df, OUTPUT_DIR)

    # ── Per-year breakdown ────────────────────────────────────────────────────
    print("\nRunning per-year breakdown with best params ...")
    _yearly_breakdown(best, cfg.api_key, DATA_DIR, OUTPUT_DIR)

    print(f"\nAll param sweep outputs in: {OUTPUT_DIR}")
    print("\n" + "=" * 65)
    print("NEXT STEPS")
    print("=" * 65)
    print(f"1. Review heatmaps in {OUTPUT_DIR}")
    print("2. Confirm best params are on a broad Sharpe plateau (not a spike)")
    print("3. Edit run_full_backtest.py:")
    print(f"     START_DATE = '2018-01-01'")
    print(f"     entry_price_edge = {best['entry_price_edge']}")
    print(f"     entry_iv_edge    = {best['entry_iv_edge']}")
    print(f"     max_holding_days = {int(best['max_holding_days'])}")
    print("4. Run OOS (2022-2024) first, then full 7-year backtest")


if __name__ == "__main__":
    main()
