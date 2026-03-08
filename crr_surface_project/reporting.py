"""
reporting.py
Institutional-quality PDF report generator for the CRR Surface Strategy.
Uses matplotlib PdfPages — no external binary dependencies.
"""
from __future__ import annotations

import math
from datetime import date as _date
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────────────────────────────────────

PAGE_SIZE = (11, 8.5)          # Letter landscape
DPI = 150

DARK_BLUE = "#1a3a5c"
MID_BLUE = "#2e6da4"
LIGHT_BLUE = "#d4e6f1"
GREEN = "#27ae60"
RED = "#c0392b"
ORANGE = "#e67e22"
GRAY = "#7f8c8d"
LIGHT_GRAY = "#ecf0f1"
TEXT_COLOR = "#2c3e50"

TITLE_FONT = {"fontsize": 16, "fontweight": "bold", "color": DARK_BLUE}
SUBTITLE_FONT = {"fontsize": 11, "color": GRAY}
SECTION_FONT = {"fontsize": 13, "fontweight": "bold", "color": DARK_BLUE}
BODY_FONT = {"fontsize": 9, "color": TEXT_COLOR}

DISCLAIMER = (
    "IMPORTANT DISCLAIMER: This document is prepared for research and educational purposes only. "
    "It does not constitute investment advice, a solicitation, or an offer to buy or sell any security. "
    "Past performance of a simulated/backtested strategy is not indicative of future results. "
    "All results are hypothetical and do not reflect actual trading. "
    "No representation is made that any account will achieve similar results."
)


def _apply_page_style(fig: plt.Figure):
    fig.patch.set_facecolor("white")


def _header_bar(fig: plt.Figure, title: str, subtitle: str = ""):
    """Adds a thin blue header band at the top of a figure."""
    ax_h = fig.add_axes([0, 0.93, 1, 0.07])
    ax_h.set_facecolor(DARK_BLUE)
    ax_h.axis("off")
    ax_h.text(0.015, 0.5, title, va="center", ha="left",
              color="white", fontsize=13, fontweight="bold", transform=ax_h.transAxes)
    if subtitle:
        ax_h.text(0.985, 0.5, subtitle, va="center", ha="right",
                  color=LIGHT_BLUE, fontsize=9, transform=ax_h.transAxes)


def _footer(fig: plt.Figure, page_note: str = ""):
    """Adds a thin footer with disclaimer snippet."""
    ax_f = fig.add_axes([0, 0, 1, 0.03])
    ax_f.set_facecolor(LIGHT_GRAY)
    ax_f.axis("off")
    text = "CRR Surface Delta-Hedge Strategy | Research Report | For informational purposes only"
    if page_note:
        text = page_note + "  |  " + text
    ax_f.text(0.5, 0.5, text, va="center", ha="center",
              color=GRAY, fontsize=7, transform=ax_f.transAxes)


def _render_table(ax: plt.Axes, df: pd.DataFrame, title: str = "",
                  col_widths: Optional[List[float]] = None,
                  fmt_funcs: Optional[Dict] = None):
    """Render a DataFrame as a styled matplotlib table."""
    ax.axis("off")
    if df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, color=GRAY, fontsize=10)
        return

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=DARK_BLUE, pad=8)

    headers = list(df.columns)
    n_cols = len(headers)
    n_rows = len(df)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    cell_text = []
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            val = row[col]
            if fmt_funcs and col in fmt_funcs:
                cells.append(fmt_funcs[col](val))
            elif isinstance(val, float):
                if np.isnan(val):
                    cells.append("—")
                elif abs(val) >= 1000:
                    cells.append(f"${val:,.0f}" if col.lower() in ("pnl", "total_pnl", "drawdown",
                                                                    "max_drawdown", "profit") else f"{val:,.1f}")
                else:
                    cells.append(f"{val:.3f}")
            elif val is None:
                cells.append("—")
            else:
                cells.append(str(val))
        cell_text.append(cells)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row_idx == 0:
            cell.set_facecolor(DARK_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        else:
            cell.set_facecolor("white")


# ─────────────────────────────────────────────────────────────────────────────
# Individual page generators
# ─────────────────────────────────────────────────────────────────────────────

def _page_cover(report_date: str, date_range_str: str = "2022-2024") -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)

    # Top color band
    ax_top = fig.add_axes([0, 0.82, 1, 0.18])
    ax_top.set_facecolor(DARK_BLUE)
    ax_top.axis("off")
    ax_top.text(0.05, 0.72, "QUANTITATIVE RESEARCH REPORT",
                color=LIGHT_BLUE, fontsize=11, fontweight="bold", transform=ax_top.transAxes)
    ax_top.text(0.05, 0.35, "CRR Binomial Surface Delta-Hedge Strategy",
                color="white", fontsize=22, fontweight="bold", transform=ax_top.transAxes)
    ax_top.text(0.05, 0.08, f"SPY Options | {date_range_str} Backtest | Institutional Research Edition",
                color=LIGHT_BLUE, fontsize=12, transform=ax_top.transAxes)

    # Body
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.72])
    ax.axis("off")

    # Description box
    desc = (
        "EXECUTIVE OVERVIEW\n\n"
        "This report presents a systematic, quantitative analysis of a Cox-Ross-Rubinstein (CRR) "
        "binomial pricing model applied to SPY options. The strategy identifies mispricings between "
        "CRR fair values and market prices using a cross-sectional implied volatility surface, "
        "entering positions when edges exceed defined thresholds and hedging delta exposure "
        "continuously via the underlying ETF.\n\n"
        "KEY HIGHLIGHTS\n\n"
        "  - Strategy Universe:  SPY options (CBOE-listed), two distinct moneyness/DTE buckets\n"
        "  - Signal Generation:  CRR binomial fair value vs. market mid price residual\n"
        "  - Risk Management:    Delta-neutral hedging, max holding period, spread filters\n"
        f"  - Backtest Period:    {date_range_str}\n"
        "  - Data Source:        Polygon.io daily OHLC aggregates\n"
        "  - Model:              American CRR binomial tree (N=80 steps, Numba-compiled)"
    )
    ax.text(0.0, 0.95, desc, va="top", ha="left", color=TEXT_COLOR,
            fontsize=9.5, linespacing=1.6, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT_BLUE, alpha=0.35, edgecolor=MID_BLUE))

    # Date
    ax.text(0.0, 0.02, f"Report generated: {report_date}",
            color=GRAY, fontsize=9, transform=ax.transAxes)

    # Disclaimer
    ax_d = fig.add_axes([0.05, 0.03, 0.90, 0.06])
    ax_d.axis("off")
    ax_d.text(0.0, 0.5, DISCLAIMER, va="center", ha="left", color=GRAY,
              fontsize=6.5, wrap=True, transform=ax_d.transAxes,
              bbox=dict(boxstyle="round", facecolor="#fff9e6", alpha=0.8, edgecolor=ORANGE))

    return fig


def _page_exec_summary(metrics: Dict) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Executive Summary", "Risk-Adjusted Performance Overview")
    _footer(fig, "Page 2")

    axes_area = fig.add_axes([0.03, 0.06, 0.94, 0.85])
    axes_area.axis("off")

    # Build summary table from all buckets
    rows_data = []
    for label, m in metrics.items():
        ov = m.get("overview", {})
        ra = m.get("risk_adjusted", {})
        bm = m.get("benchmark", {})
        dd = m.get("drawdown", {})
        tr = m.get("trades", {})
        at = m.get("alpha_test", {})

        def _fmt(v, prefix="", suffix="", pct=False):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "—"
            if pct:
                return f"{v*100:.1f}%"
            return f"{prefix}{v:,.2f}{suffix}"

        rows_data.append([
            label.replace("_", " "),
            _fmt(ov.get("total_pnl"), prefix="$"),
            _fmt(ra.get("sharpe")),
            _fmt(ra.get("sortino")),
            _fmt(ra.get("calmar")),
            _fmt(dd.get("max_drawdown"), prefix="$"),
            _fmt(tr.get("win_rate"), pct=True),
            _fmt(bm.get("information_ratio")),
            _fmt(at.get("annualized_alpha"), prefix="$"),
            f"{at.get('p_value', float('nan')):.3f}" if isinstance(at.get('p_value'), float) and math.isfinite(at.get('p_value', float('nan'))) else "—",
        ])

    col_labels = ["Strategy", "Total P&L", "Sharpe", "Sortino", "Calmar",
                  "Max DD", "Win Rate", "IR vs SPY", "Ann. Alpha", "α p-value"]

    df_summary = pd.DataFrame(rows_data, columns=col_labels)
    first_ov = list(metrics.values())[0].get("overview", {}) if metrics else {}
    start_yr = str(first_ov.get("start_date", ""))[:4]
    end_yr = str(first_ov.get("end_date", ""))[:4]
    period_str = f"{start_yr}-{end_yr}" if start_yr and end_yr else "Full Period"
    _render_table(axes_area, df_summary, title=f"Strategy Performance Summary  ({period_str})")

    return fig


def _page_equity_curve(equity_dict: Dict[str, pd.DataFrame], spy_prices: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Equity Curve & Daily P&L", "Cumulative Strategy P&L vs. SPY Buy-and-Hold")
    _footer(fig, "Page 3")

    gs = gridspec.GridSpec(3, 1, top=0.90, bottom=0.06, left=0.07, right=0.97,
                           hspace=0.05, height_ratios=[3, 1.2, 1.2])

    # Top panel: cumulative P&L
    ax1 = fig.add_subplot(gs[0])
    colors_map = {0: MID_BLUE, 1: ORANGE}

    for i, (label, equity) in enumerate(equity_dict.items()):
        if equity.empty:
            continue
        df = equity.copy()
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        ax1.plot(df.index, df["cum_pnl"], color=list(colors_map.values())[i],
                 linewidth=1.8, label=label.replace("_", " "))

    # SPY buy-and-hold benchmark
    if not spy_prices.empty:
        from extended_metrics import spy_buy_hold_pnl
        spy_pnl = spy_buy_hold_pnl(spy_prices)
        spy_cum = spy_pnl.cumsum()
        ax1.plot(spy_cum.index, spy_cum.values, color=GRAY, linewidth=1.2,
                 linestyle="--", alpha=0.7, label="SPY Buy-and-Hold")

    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax1.set_ylabel("Cumulative P&L ($)", fontsize=9, color=TEXT_COLOR)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_facecolor("#f8f9fa")
    ax1.tick_params(axis="x", labelsize=7)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.grid(True, alpha=0.4, linewidth=0.5)
    ax1.set_xticklabels([])

    # Middle panel: daily P&L bars (best bucket)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    best_equity = list(equity_dict.values())[0]
    if not best_equity.empty:
        df2 = best_equity.copy()
        if "date" in df2.columns:
            df2 = df2.set_index("date")
        df2.index = pd.to_datetime(df2.index)
        pos_mask = df2["daily_pnl"] >= 0
        ax2.bar(df2.index[pos_mask], df2["daily_pnl"][pos_mask],
                color=GREEN, width=1.5, alpha=0.7, label="Daily P&L (positive)")
        ax2.bar(df2.index[~pos_mask], df2["daily_pnl"][~pos_mask],
                color=RED, width=1.5, alpha=0.7, label="Daily P&L (negative)")
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_ylabel("Daily P&L ($)", fontsize=8)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax2.set_facecolor("#f8f9fa")
        ax2.tick_params(axis="x", labelsize=7)
        ax2.tick_params(axis="y", labelsize=7)
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        ax2.set_title(f"Daily P&L: {list(equity_dict.keys())[0].replace('_', ' ')}",
                      fontsize=8, loc="left", color=GRAY)

    # Bottom panel: open positions
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if not best_equity.empty:
        df3 = best_equity.copy()
        if "date" in df3.columns:
            df3 = df3.set_index("date")
        ax3.fill_between(df3.index, df3["open_positions"], alpha=0.5,
                         color=MID_BLUE, step="mid")
        ax3.set_ylabel("Open\nPositions", fontsize=8)
        ax3.set_ylim(0, df3["open_positions"].max() + 1)
        ax3.set_facecolor("#f8f9fa")
        ax3.tick_params(axis="x", labelsize=7)
        ax3.tick_params(axis="y", labelsize=7)
        ax3.grid(True, alpha=0.3, linewidth=0.5)

    return fig


def _page_drawdown(equity_dict: Dict[str, pd.DataFrame]) -> plt.Figure:
    from extended_metrics import drawdown_series, drawdown_table

    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Drawdown Analysis", "Underwater Curve & Top Drawdown Episodes")
    _footer(fig, "Page 4")

    gs = gridspec.GridSpec(2, 2, top=0.90, bottom=0.06, left=0.07, right=0.97,
                           hspace=0.4, wspace=0.3)

    colors_list = [MID_BLUE, ORANGE]

    for col_idx, (label, equity) in enumerate(equity_dict.items()):
        if equity.empty or col_idx >= 2:
            continue

        df = equity.copy()
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        cum = df["cum_pnl"].fillna(0.0)
        dd = drawdown_series(cum)

        # Underwater curve
        ax1 = fig.add_subplot(gs[0, col_idx])
        ax1.fill_between(dd.index, dd.values, 0, alpha=0.5,
                         color=colors_list[col_idx])
        ax1.plot(dd.index, dd.values, color=colors_list[col_idx], linewidth=0.8)
        ax1.set_title(f"Drawdown: {label.replace('_', ' ')}", fontsize=9,
                      fontweight="bold", color=DARK_BLUE)
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.set_facecolor("#f8f9fa")
        ax1.tick_params(labelsize=7)
        ax1.grid(True, alpha=0.4, linewidth=0.5)

        # Top-5 drawdown table
        ax2 = fig.add_subplot(gs[1, col_idx])
        dd_tbl = drawdown_table(equity, top_n=5)
        if not dd_tbl.empty:
            display_cols = ["start_date", "trough_date", "max_drawdown", "duration_days"]
            dd_display = dd_tbl[display_cols].copy()
            dd_display.columns = ["Start", "Trough", "Max DD ($)", "Days"]
            dd_display["Max DD ($)"] = dd_display["Max DD ($)"].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
            )
            dd_display["Start"] = pd.to_datetime(dd_display["Start"]).dt.strftime("%Y-%m-%d")
            dd_display["Trough"] = pd.to_datetime(dd_display["Trough"]).dt.strftime("%Y-%m-%d")
            _render_table(ax2, dd_display, title="Top Drawdown Episodes")

    return fig


def _page_rolling_metrics(equity_dict: Dict[str, pd.DataFrame]) -> plt.Figure:
    from extended_metrics import rolling_sharpe, rolling_volatility, rolling_returns

    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Rolling Risk Metrics", "30-Day Rolling Sharpe, Volatility, and Returns")
    _footer(fig, "Page 5")

    n_buckets = min(len(equity_dict), 2)
    gs = gridspec.GridSpec(3, n_buckets, top=0.90, bottom=0.06, left=0.08, right=0.97,
                           hspace=0.35, wspace=0.25)

    colors_per_bucket = [MID_BLUE, ORANGE]

    for col_idx, (label, equity) in enumerate(equity_dict.items()):
        if col_idx >= 2 or equity.empty:
            continue

        df = equity.copy()
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        pnl = df["daily_pnl"].fillna(0.0)
        bucket_color = colors_per_bucket[col_idx]

        # Rolling Sharpe
        ax1 = fig.add_subplot(gs[0, col_idx])
        rs = rolling_sharpe(pnl, window=30)
        ax1.plot(pnl.index, rs.values, color=bucket_color, linewidth=1.2)
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.axhline(2.0, color=GREEN, linewidth=0.8, linestyle="--", alpha=0.7, label="Sharpe=2.0")
        ax1.fill_between(pnl.index, rs.values, 0,
                         where=(rs.values > 0), alpha=0.2, color=GREEN)
        ax1.fill_between(pnl.index, rs.values, 0,
                         where=(rs.values <= 0), alpha=0.2, color=RED)
        ax1.set_ylabel("30-Day Rolling Sharpe", fontsize=8)
        ax1.legend(fontsize=7)
        ax1.set_facecolor("#f8f9fa")
        ax1.tick_params(labelsize=7)
        ax1.grid(True, alpha=0.4, linewidth=0.5)
        ax1.set_title(f"Rolling Metrics: {label.replace('_', ' ')}", fontsize=8,
                      fontweight="bold", color=DARK_BLUE, loc="left")

        # Rolling volatility
        ax2 = fig.add_subplot(gs[1, col_idx], sharex=ax1)
        rv = rolling_volatility(pnl, window=30)
        ax2.plot(pnl.index, rv.values, color=ORANGE, linewidth=1.2)
        ax2.set_ylabel("Ann. Volatility ($)", fontsize=8)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax2.set_facecolor("#f8f9fa")
        ax2.tick_params(labelsize=7)
        ax2.grid(True, alpha=0.4, linewidth=0.5)

        # Rolling 21-day returns
        ax3 = fig.add_subplot(gs[2, col_idx], sharex=ax1)
        rr = rolling_returns(pnl, window=21)
        pos = rr.values >= 0
        ax3.bar(pnl.index[pos], rr.values[pos], color=GREEN, alpha=0.6, width=1)
        ax3.bar(pnl.index[~pos], rr.values[~pos], color=RED, alpha=0.6, width=1)
        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_ylabel("21-Day Return ($)", fontsize=8)
        ax3.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax3.set_facecolor("#f8f9fa")
        ax3.tick_params(labelsize=7)
        ax3.grid(True, alpha=0.4, linewidth=0.5)

    return fig


def _page_monthly_pnl(metrics: Dict) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Monthly P&L Breakdown", "P&L by Calendar Month and Year")
    _footer(fig, "Page 6")

    gs = gridspec.GridSpec(1, 2, top=0.90, bottom=0.06, left=0.04, right=0.97, wspace=0.1)

    for col_idx, (label, m) in enumerate(metrics.items()):
        if col_idx >= 2:
            break
        monthly = m.get("monthly")
        if monthly is None or (isinstance(monthly, pd.DataFrame) and monthly.empty):
            continue

        ax = fig.add_subplot(gs[col_idx])
        ax.axis("off")

        monthly_df = monthly.copy()
        monthly_df = monthly_df.fillna(0.0)

        # Color cells by sign/magnitude
        n_rows, n_cols = monthly_df.shape
        col_labels_list = list(monthly_df.columns)
        row_labels_list = [str(y) for y in monthly_df.index]

        cell_text = []
        cell_colors = []
        for r_idx in range(n_rows):
            row_text = []
            row_colors = []
            for c_idx, col in enumerate(col_labels_list):
                val = monthly_df.iloc[r_idx, c_idx]
                row_text.append(f"${val:,.0f}")
                if col == "Annual":
                    intensity = min(abs(val) / max(abs(monthly_df["Annual"].max()), 1), 1.0)
                    color = mcolors.to_rgba(GREEN, alpha=0.3 + 0.5 * intensity) if val >= 0 \
                        else mcolors.to_rgba(RED, alpha=0.3 + 0.5 * intensity)
                else:
                    max_abs = max(abs(monthly_df[monthly_df.columns.drop("Annual")].values.max()), 1)
                    intensity = min(abs(val) / max_abs, 1.0)
                    color = mcolors.to_rgba(GREEN, alpha=0.15 + 0.45 * intensity) if val >= 0 \
                        else mcolors.to_rgba(RED, alpha=0.15 + 0.45 * intensity)
                row_colors.append(color)
            cell_text.append(row_text)
            cell_colors.append(row_colors)

        tbl = ax.table(
            cellText=cell_text,
            rowLabels=row_labels_list,
            colLabels=col_labels_list,
            cellColours=cell_colors,
            loc="center",
            cellLoc="right",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)

        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(DARK_BLUE)
                cell.set_text_props(color="white", fontweight="bold", fontsize=7)
            cell.set_edgecolor("#dddddd")

        ax.set_title(f"Monthly P&L — {label.replace('_', ' ')}",
                     fontsize=10, fontweight="bold", color=DARK_BLUE, pad=10)

    return fig


def _page_yearly_summary(metrics: Dict) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Annual Performance Summary", "Year-by-Year Breakdown")
    _footer(fig, "Page 7")

    gs = gridspec.GridSpec(2, 2, top=0.90, bottom=0.06, left=0.07, right=0.97,
                           hspace=0.45, wspace=0.35)

    for col_idx, (label, m) in enumerate(metrics.items()):
        if col_idx >= 2:
            break

        yearly = m.get("yearly")
        if yearly is None or (isinstance(yearly, pd.DataFrame) and yearly.empty):
            continue

        yearly_df = yearly.copy()

        # Bar chart of annual P&L
        ax1 = fig.add_subplot(gs[0, col_idx])
        years = yearly_df["year"].astype(str)
        pnls = yearly_df["total_pnl"]
        bar_colors = [GREEN if v >= 0 else RED for v in pnls]
        bars = ax1.bar(years, pnls, color=bar_colors, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax1.axhline(0, color="black", linewidth=0.6)
        ax1.set_ylabel("Annual P&L ($)", fontsize=9)
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.set_title(f"Annual P&L: {label.replace('_', ' ')}", fontsize=9,
                      fontweight="bold", color=DARK_BLUE)
        ax1.set_facecolor("#f8f9fa")
        ax1.tick_params(labelsize=8)
        ax1.grid(True, alpha=0.4, linewidth=0.5, axis="y")
        for bar, val in zip(bars, pnls):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + abs(pnls.max()) * 0.02,
                     f"${val:,.0f}", ha="center", fontsize=7.5, color=TEXT_COLOR)

        # Metrics table
        ax2 = fig.add_subplot(gs[1, col_idx])
        display_df = yearly_df[["year", "total_pnl", "sharpe", "sortino", "max_drawdown", "win_rate"]].copy()
        display_df.columns = ["Year", "P&L ($)", "Sharpe", "Sortino", "Max DD ($)", "Win Rate"]
        display_df["P&L ($)"] = display_df["P&L ($)"].map(lambda x: f"${x:,.0f}")
        display_df["Max DD ($)"] = display_df["Max DD ($)"].map(lambda x: f"${x:,.0f}")
        display_df["Sharpe"] = display_df["Sharpe"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "—")
        display_df["Sortino"] = display_df["Sortino"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "—")
        display_df["Win Rate"] = display_df["Win Rate"].map(lambda x: f"{x*100:.1f}%")
        _render_table(ax2, display_df)

    return fig


def _page_benchmark_comparison(metrics: Dict, spy_prices: pd.DataFrame,
                                equity_dict: Dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Benchmark Comparison", "Strategy vs. SPY Buy-and-Hold")
    _footer(fig, "Page 8")

    gs = gridspec.GridSpec(2, 2, top=0.90, bottom=0.06, left=0.07, right=0.97,
                           hspace=0.45, wspace=0.3)

    # Grouped bar chart: Sharpe + Sortino + Total Return
    ax1 = fig.add_subplot(gs[0, :])
    labels_list = list(metrics.keys())
    metrics_to_plot = ["strategy_sharpe", "benchmark_sharpe", "strategy_sortino", "benchmark_sortino"]
    labels_display = ["Strat Sharpe", "SPY Sharpe", "Strat Sortino", "SPY Sortino"]

    x = np.arange(len(labels_list))
    width = 0.18
    colors_bars = [MID_BLUE, LIGHT_GRAY, ORANGE, LIGHT_GRAY]

    for i, (metric_key, disp_label, color) in enumerate(zip(metrics_to_plot, labels_display, colors_bars)):
        vals = []
        for label in labels_list:
            bm = metrics[label].get("benchmark", {})
            v = bm.get(metric_key, float("nan"))
            vals.append(float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0.0)
        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, vals, width, label=disp_label, color=color,
                       edgecolor="white", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace("_", "\n") for l in labels_list], fontsize=8)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Ratio", fontsize=9)
    ax1.set_title("Risk-Adjusted Return Comparison: Strategy vs. SPY Buy-and-Hold", fontsize=9,
                  fontweight="bold", color=DARK_BLUE)
    ax1.legend(fontsize=8, ncol=4)
    ax1.set_facecolor("#f8f9fa")
    ax1.tick_params(labelsize=8)
    ax1.grid(True, alpha=0.4, linewidth=0.5, axis="y")

    # Alpha scatter for best bucket
    ax2 = fig.add_subplot(gs[1, 0])
    best_label = list(equity_dict.keys())[0]
    best_equity = equity_dict[best_label]
    if not best_equity.empty and not spy_prices.empty:
        df = best_equity.copy()
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

        spy_closes = spy_prices["close"].sort_index()
        spy_ret = spy_closes.pct_change().reindex(df.index, fill_value=0.0)

        pnl = df["daily_pnl"].fillna(0.0)
        valid = (spy_ret.abs() < 0.1) & (pnl.abs() < pnl.std() * 5)

        ax2.scatter(spy_ret[valid], pnl[valid], alpha=0.3, s=8, color=MID_BLUE)

        # Regression line
        x_arr = spy_ret[valid].values
        y_arr = pnl[valid].values
        if len(x_arr) > 5 and np.std(x_arr) > 0:
            coeffs = np.polyfit(x_arr, y_arr, 1)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
            ax2.plot(x_line, np.polyval(coeffs, x_line), color=RED, linewidth=1.5,
                     label=f"β={coeffs[0]:,.0f}, α={coeffs[1]:+.2f}/day")

        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.axvline(0, color="black", linewidth=0.5)
        ax2.set_xlabel("SPY Daily Return", fontsize=9)
        ax2.set_ylabel("Strategy Daily P&L ($)", fontsize=9)
        ax2.set_title(f"Alpha Scatter: {best_label.replace('_', ' ')}", fontsize=9,
                      fontweight="bold", color=DARK_BLUE)
        ax2.legend(fontsize=8)
        ax2.set_facecolor("#f8f9fa")
        ax2.tick_params(labelsize=7)
        ax2.grid(True, alpha=0.4, linewidth=0.5)

    # Benchmark metrics table
    ax3 = fig.add_subplot(gs[1, 1])
    rows = []
    for label in labels_list:
        bm = metrics[label].get("benchmark", {})
        def _f(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "—"
            return f"{v:.3f}"
        rows.append([
            label.replace("_", " ")[:20],
            f"${bm.get('strategy_total_pnl', 0):,.0f}",
            f"${bm.get('benchmark_total_pnl', 0):,.0f}",
            _f(bm.get("correlation_with_spy")),
            _f(bm.get("beta_to_spy")),
            _f(bm.get("information_ratio")),
        ])
    df_bm = pd.DataFrame(rows, columns=["Strategy", "Strat P&L", "SPY P&L", "Corr", "Beta", "IR"])
    _render_table(ax3, df_bm, title="Benchmark Metrics Summary")

    return fig


def _page_alpha_significance(metrics: Dict) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Statistical Significance of Alpha", "T-Test: H₀ = Mean Daily P&L is Zero")
    _footer(fig, "Page 9")

    ax = fig.add_axes([0.05, 0.06, 0.90, 0.82])
    ax.axis("off")

    rows = []
    for label, m in metrics.items():
        at = m.get("alpha_test", {})
        ov = m.get("overview", {})

        def _f(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "—"
            return f"{v:.4f}"

        sig_str = "Yes ✓" if at.get("is_significant_5pct") else "No"
        rows.append([
            label.replace("_", " "),
            str(ov.get("trading_days", "—")),
            f"${ov.get('mean_daily_pnl', 0):,.2f}",
            f"${at.get('annualized_alpha', 0):,.0f}",
            _f(at.get("t_stat")),
            _f(at.get("p_value")),
            sig_str,
        ])

    df_sig = pd.DataFrame(rows, columns=[
        "Strategy", "N (days)", "Mean Daily P&L", "Ann. Alpha", "t-Statistic", "p-value", "Sig. @ 5%"
    ])
    _render_table(ax, df_sig, title="Alpha Statistical Significance (Two-Sided t-Test, H₀: μ = 0)")

    # Explanation text
    fig.text(0.07, 0.12,
             "Interpretation: A p-value < 0.05 (two-tailed) indicates the strategy generates statistically\n"
             "significant non-zero daily P&L, i.e. the observed alpha is unlikely due to chance alone.\n"
             "Note: Autocorrelation in daily P&L may inflate t-statistics; interpret with caution.",
             fontsize=9, color=GRAY, va="bottom",
             bbox=dict(boxstyle="round", facecolor="#fff9e6", alpha=0.7, edgecolor=ORANGE))

    return fig


def _page_trade_statistics(metrics: Dict) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Trade-Level Analytics", "Per-Trade P&L Distribution & Statistics")
    _footer(fig, "Page 10")

    gs = gridspec.GridSpec(2, 3, top=0.90, bottom=0.06, left=0.06, right=0.97,
                           hspace=0.5, wspace=0.35)

    for col_offset, (label, m) in enumerate(metrics.items()):
        if col_offset >= 2:
            break

        trade_detail = m.get("trade_detail")
        trade_stats = m.get("trades", {})

        # P&L histogram
        ax1 = fig.add_subplot(gs[0, col_offset])
        if isinstance(trade_detail, pd.DataFrame) and not trade_detail.empty:
            pnl_vals = trade_detail["trade_pnl"].dropna()
            if len(pnl_vals) > 0:
                ax1.hist(pnl_vals, bins=min(30, max(len(pnl_vals) // 3, 5)),
                         color=MID_BLUE, alpha=0.7, edgecolor="white", linewidth=0.5)
                ax1.axvline(0, color=RED, linewidth=1.2, linestyle="--")
                ax1.axvline(float(pnl_vals.mean()), color=GREEN, linewidth=1.2,
                            linestyle="--", label=f"Mean: ${pnl_vals.mean():,.2f}")
                ax1.set_xlabel("Trade P&L ($)", fontsize=8)
                ax1.set_ylabel("Frequency", fontsize=8)
                ax1.legend(fontsize=7)
        ax1.set_title(f"P&L Dist: {label[:20].replace('_', ' ')}", fontsize=8,
                      fontweight="bold", color=DARK_BLUE)
        ax1.set_facecolor("#f8f9fa")
        ax1.tick_params(labelsize=7)
        ax1.grid(True, alpha=0.3, linewidth=0.5)

        # Win/loss pie
        ax2 = fig.add_subplot(gs[1, col_offset])
        if isinstance(trade_detail, pd.DataFrame) and not trade_detail.empty:
            pnl_vals = trade_detail["trade_pnl"].dropna()
            n_win = int((pnl_vals > 0).sum())
            n_loss = int((pnl_vals <= 0).sum())
            if n_win + n_loss > 0:
                wedge_colors = [GREEN, RED]
                ax2.pie([n_win, n_loss], labels=[f"Win ({n_win})", f"Loss ({n_loss})"],
                        colors=wedge_colors, autopct="%1.1f%%",
                        startangle=90, textprops={"fontsize": 8})
        ax2.set_title("Win/Loss", fontsize=8, fontweight="bold", color=DARK_BLUE)

    # Trade stats table (full width, bottom)
    ax3 = fig.add_subplot(gs[:, 2])
    all_rows = []
    for label, m in metrics.items():
        ts = m.get("trades", {})
        def _f(v):
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return "—"
            return f"{v:.2f}"
        all_rows.append(["Metric", label.replace("_", " ")[:18]])
        all_rows.append(["Total Entries", str(ts.get("total_entries", "—"))])
        all_rows.append(["Long Entries", str(ts.get("long_entries", "—"))])
        all_rows.append(["Short Entries", str(ts.get("short_entries", "—"))])
        all_rows.append(["Win Rate", f"{ts.get('win_rate', 0)*100:.1f}%" if ts.get('win_rate') else "—"])
        all_rows.append(["Avg Holding Days", _f(ts.get("avg_holding_days"))])
        all_rows.append(["Profit Factor", _f(ts.get("profit_factor"))])
        all_rows.append(["Avg Winner ($)", f"${ts.get('avg_winner_pnl', 0):,.2f}" if ts.get('avg_winner_pnl') else "—"])
        all_rows.append(["Avg Loser ($)", f"${ts.get('avg_loser_pnl', 0):,.2f}" if ts.get('avg_loser_pnl') else "—"])
        all_rows.append(["Largest Win ($)", f"${ts.get('largest_win', 0):,.2f}" if ts.get('largest_win') else "—"])
        all_rows.append(["Largest Loss ($)", f"${ts.get('largest_loss', 0):,.2f}" if ts.get('largest_loss') else "—"])
        all_rows.append(["", ""])

    if all_rows:
        ax3.axis("off")
        tbl = ax3.table(
            cellText=all_rows,
            loc="center", cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        for (r, c), cell in tbl.get_celld().items():
            if all_rows[r][0] == "Metric":
                cell.set_facecolor(DARK_BLUE)
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f0f4f8")
            cell.set_edgecolor("#cccccc")

    return fig


def _page_bucket_comparison(equity_dict: Dict[str, pd.DataFrame],
                             metrics: Dict) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Bucket Strategy Comparison", "PUT_30_60_OTM vs. CALL_120_150_ATM")
    _footer(fig, "Page 11")

    gs = gridspec.GridSpec(2, 1, top=0.90, bottom=0.06, left=0.08, right=0.97,
                           hspace=0.4, height_ratios=[2, 1])

    # Dual equity curves
    ax1 = fig.add_subplot(gs[0])
    colors_map = [MID_BLUE, ORANGE]
    for i, (label, equity) in enumerate(equity_dict.items()):
        if equity.empty:
            continue
        df = equity.copy()
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        ax1.plot(df.index, df["cum_pnl"], color=colors_map[i % 2],
                 linewidth=1.8, label=label.replace("_", " "))

    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Cumulative P&L ($)", fontsize=9)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(fontsize=9)
    ax1.set_facecolor("#f8f9fa")
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.4, linewidth=0.5)
    ax1.set_title("Cumulative P&L: Best Bucket vs. Worst Bucket", fontsize=10,
                  fontweight="bold", color=DARK_BLUE)

    # Comparison table
    ax2 = fig.add_subplot(gs[1])
    rows = []
    for label, m in metrics.items():
        ov = m.get("overview", {})
        ra = m.get("risk_adjusted", {})
        dd = m.get("drawdown", {})
        tr = m.get("trades", {})
        def _f(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "—"
            return f"{v:.2f}"
        rows.append([
            label.replace("_", " "),
            f"${ov.get('total_pnl', 0):,.0f}",
            _f(ra.get("sharpe")),
            _f(ra.get("sortino")),
            f"${dd.get('max_drawdown', 0):,.0f}",
            f"{tr.get('win_rate', 0)*100:.1f}%" if tr.get('win_rate') else "—",
            str(tr.get("total_entries", "—")),
        ])
    df_comp = pd.DataFrame(rows,
        columns=["Strategy", "Total P&L", "Sharpe", "Sortino", "Max DD", "Win Rate", "Trades"])
    _render_table(ax2, df_comp, title="Side-by-Side Performance Comparison")

    return fig


def _page_regime_analysis(metrics: Dict, spy_prices: pd.DataFrame,
                           equity_dict: Dict[str, pd.DataFrame]) -> plt.Figure:
    from extended_metrics import trend_regime_breakdown

    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Market Regime Analysis", "Performance by VIX Regime and SPY Trend Regime")
    _footer(fig, "Page 12")

    gs = gridspec.GridSpec(2, 2, top=0.90, bottom=0.06, left=0.07, right=0.97,
                           hspace=0.45, wspace=0.3)

    best_label = list(equity_dict.keys())[0]
    best_equity = equity_dict[best_label]
    best_metrics = list(metrics.values())[0]

    # VIX regime (if available)
    ax1 = fig.add_subplot(gs[0, 0])
    vix_regime = best_metrics.get("vix_regime")
    if isinstance(vix_regime, pd.DataFrame) and not vix_regime.empty:
        ax1.bar(vix_regime["regime"], vix_regime["total_pnl"],
                color=[GREEN if v >= 0 else RED for v in vix_regime["total_pnl"]],
                alpha=0.8, edgecolor="white")
        ax1.set_title("P&L by VIX Regime", fontsize=9, fontweight="bold", color=DARK_BLUE)
        ax1.set_ylabel("Total P&L ($)", fontsize=8)
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.set_facecolor("#f8f9fa")
        ax1.tick_params(labelsize=7)
        ax1.grid(True, alpha=0.4, linewidth=0.5, axis="y")
    else:
        ax1.text(0.5, 0.5, "VIX data not available\n(fetch I:VIX from Polygon)", ha="center",
                 va="center", color=GRAY, fontsize=9, transform=ax1.transAxes)
        ax1.axis("off")
        ax1.set_title("P&L by VIX Regime", fontsize=9, fontweight="bold", color=DARK_BLUE)

    # Trend regime
    ax2 = fig.add_subplot(gs[0, 1])
    trend_regime = best_metrics.get("trend_regime")
    if isinstance(trend_regime, pd.DataFrame) and not trend_regime.empty:
        ax2.bar(trend_regime["regime"], trend_regime["total_pnl"],
                color=[GREEN if v >= 0 else RED for v in trend_regime["total_pnl"]],
                alpha=0.8, edgecolor="white")
        ax2.set_title("P&L by SPY Trend Regime", fontsize=9, fontweight="bold", color=DARK_BLUE)
        ax2.set_ylabel("Total P&L ($)", fontsize=8)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax2.tick_params(axis="x", labelsize=6)
        ax2.tick_params(axis="y", labelsize=7)
        ax2.set_facecolor("#f8f9fa")
        ax2.grid(True, alpha=0.4, linewidth=0.5, axis="y")

    # Regime table for best bucket
    ax3 = fig.add_subplot(gs[1, :])
    regime_rows = []
    for reg_name, regime_df in [("VIX", vix_regime), ("Trend", trend_regime)]:
        if isinstance(regime_df, pd.DataFrame) and not regime_df.empty:
            for _, r in regime_df.iterrows():
                regime_rows.append([
                    reg_name, str(r["regime"]),
                    str(r.get("days", "—")),
                    f"${r.get('total_pnl', 0):,.0f}",
                    f"{r.get('sharpe', float('nan')):.2f}" if np.isfinite(r.get('sharpe', float('nan'))) else "—",
                    f"{r.get('win_rate', 0)*100:.1f}%",
                    f"${r.get('avg_daily_pnl', 0):.2f}",
                ])

    if regime_rows:
        df_reg = pd.DataFrame(regime_rows,
            columns=["Type", "Regime", "Days", "Total P&L", "Sharpe", "Win Rate", "Avg Daily P&L"])
        _render_table(ax3, df_reg, title=f"Regime Performance Detail: {best_label.replace('_', ' ')}")
    else:
        ax3.text(0.5, 0.5, "Regime data not available", ha="center", va="center",
                 color=GRAY, fontsize=10, transform=ax3.transAxes)
        ax3.axis("off")

    return fig


def _page_convergence(convergence_results: Optional[Dict]) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "CRR Convergence Analysis", "Price & Delta Error vs. N Steps | Accuracy–Speed Trade-off")
    _footer(fig, "Page 13")

    if not convergence_results:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.text(0.5, 0.5, "Run run_convergence.py first to generate convergence data.",
                ha="center", va="center", fontsize=12, color=GRAY)
        ax.axis("off")
        return fig

    gs = gridspec.GridSpec(2, 2, top=0.90, bottom=0.06, left=0.08, right=0.97,
                           hspace=0.45, wspace=0.3)

    colors = plt.cm.tab10.colors
    RECOMMENDED_N = 80

    # Price error
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (label, data) in enumerate(convergence_results.items()):
        df = data["convergence"]
        ax1.loglog(df["steps"], df["abs_error"],
                   color=colors[i % len(colors)], linewidth=1.2, label=label[:20])
    ax1.axvline(RECOMMENDED_N, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("N steps", fontsize=8)
    ax1.set_ylabel("Price Error vs. B-S ($)", fontsize=8)
    ax1.set_title("Price Error (log-log)", fontsize=9, fontweight="bold", color=DARK_BLUE)
    ax1.legend(fontsize=6, loc="upper right")
    ax1.set_facecolor("#f8f9fa")
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.4)

    # Delta error
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (label, data) in enumerate(convergence_results.items()):
        df = data["convergence"]
        ax2.loglog(df["steps"], df["delta_abs_error"],
                   color=colors[i % len(colors)], linewidth=1.2, label=label[:20])
    ax2.axvline(RECOMMENDED_N, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("N steps", fontsize=8)
    ax2.set_ylabel("Delta Error vs. B-S", fontsize=8)
    ax2.set_title("Delta Error (log-log)", fontsize=9, fontweight="bold", color=DARK_BLUE)
    ax2.legend(fontsize=6, loc="upper right")
    ax2.set_facecolor("#f8f9fa")
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.4)

    # Time vs N
    ax3 = fig.add_subplot(gs[1, 0])
    first_label = list(convergence_results.keys())[0]
    timing_df = convergence_results[first_label]["timing"]
    ax3.loglog(timing_df["steps"], timing_df["median_time_us"],
               color=MID_BLUE, linewidth=1.5)
    ax3.fill_between(timing_df["steps"], timing_df["p25_time_us"], timing_df["p75_time_us"],
                     alpha=0.2, color=MID_BLUE)
    ax3.axvline(RECOMMENDED_N, color="black", linestyle="--", linewidth=1, alpha=0.7,
                label=f"N={RECOMMENDED_N}")
    ax3.set_xlabel("N steps", fontsize=8)
    ax3.set_ylabel("Time per Call (μs)", fontsize=8)
    ax3.set_title("Compute Time (log-log)", fontsize=9, fontweight="bold", color=DARK_BLUE)
    ax3.legend(fontsize=7)
    ax3.set_facecolor("#f8f9fa")
    ax3.tick_params(labelsize=7)
    ax3.grid(True, alpha=0.4)

    # Pareto frontier
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (label, data) in enumerate(convergence_results.items()):
        conv_df = data["convergence"]
        t_df = data["timing"]
        merged = pd.merge(conv_df[["steps", "abs_error"]],
                          t_df[["steps", "median_time_us"]], on="steps", how="inner")
        ax4.scatter(merged["median_time_us"], merged["abs_error"],
                    s=12, alpha=0.6, color=colors[i % len(colors)], label=label[:18])
        # Annotate recommended N
        row = merged[merged["steps"] == RECOMMENDED_N]
        if not row.empty:
            ax4.annotate(f"N={RECOMMENDED_N}",
                         xy=(row["median_time_us"].iloc[0], row["abs_error"].iloc[0]),
                         xytext=(5, 5), textcoords="offset points", fontsize=6)

    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Time (μs)", fontsize=8)
    ax4.set_ylabel("Price Error ($)", fontsize=8)
    ax4.set_title("Pareto Frontier", fontsize=9, fontweight="bold", color=DARK_BLUE)
    ax4.legend(fontsize=6)
    ax4.set_facecolor("#f8f9fa")
    ax4.tick_params(labelsize=7)
    ax4.grid(True, alpha=0.4)

    return fig


def _page_methodology() -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Methodology", "Model Specification, Signal Generation & Assumptions")
    _footer(fig, "Page 14")

    ax = fig.add_axes([0.05, 0.06, 0.90, 0.84])
    ax.axis("off")

    text = """
CRR BINOMIAL PRICING MODEL
──────────────────────────────────────────────────────────────────────────────
The pricing engine uses the Cox-Ross-Rubinstein (CRR) binomial tree for American-style options.

  Up factor:           u = exp(σ√Δt)           where Δt = T/N
  Down factor:         d = 1/u
  Risk-neutral prob:   p = (e^((r-q)Δt) - d) / (u - d),  clipped to [1e-9, 1-1e-9]
  Discount rate:       e^(-rΔt) per step
  American exercise:   Intrinsic value check at every node (max(hold, intrinsic))
  Delta:               (V_up - V_down) / (S_up - S_down)  [first-step difference]
  Implementation:      Numba JIT-compiled kernels for O(N²) loop performance
  Default N:           80 steps (see convergence analysis for justification)

IMPLIED VOLATILITY SURFACE
──────────────────────────────────────────────────────────────────────────────
  Cross-sectional polynomial regression, fit daily on each option chain:

      IV = β₀ + β₁·ln(K/S) + β₂·[ln(K/S)]² + β₃·√T

  where: K = strike, S = spot, T = time-to-expiry (years), fit via least squares.
  IV bounds: [0.01, 2.50]. Minimum chain size: 6 contracts for stable fit.
  IV inversion: bisection search on CRR model, tolerance 1e-5, max 100 iterations.

SIGNAL GENERATION
──────────────────────────────────────────────────────────────────────────────
  1. Fit daily IV surface to full option chain
  2. Compute CRR fair price using surface-fitted IV for each contract
  3. Signal residual:  residual = CRR_fair_price − market_mid
  4. Long signal  (cheap option):  residual ≥ +0.15  AND  IV_residual ≥ +0.003
  5. Short signal (rich option):   residual ≤ −0.15  AND  IV_residual ≤ −0.003
  Rank signals by signal_strength = |residual_price| + 10 × |residual_IV|

PORTFOLIO CONSTRUCTION & RISK MANAGEMENT
──────────────────────────────────────────────────────────────────────────────
  Max open positions:   5 concurrent
  Delta hedge:          –side × contracts × 100 × Δ shares of underlying
  Rehedge threshold:    4 shares deviation before rebalancing
  Max holding period:   10 calendar days
  Near-expiry exit:     Close if DTE < 7
  Spread filter:        Reject contracts where bid-ask spread > 60% of mid price
  Volume filter:        Minimum 1 contract traded per day

DATA & ASSUMPTIONS
──────────────────────────────────────────────────────────────────────────────
  Data source:          Polygon.io REST API (daily OHLC aggregates)
  Bid-ask synthesis:    Assumed 25 bps spread; bid = mid×(1-12.5bps), ask = mid×(1+12.5bps)
  Risk-free rate:       4.5% (constant; approximate SOFR/T-bill for backtest period)
  Dividend yield:       1.2% (SPY historical dividend yield approximation)
  Universe:             SPY options only (two pre-defined moneyness/DTE buckets)
  No transaction costs beyond bid-ask spread are modeled.
    """

    ax.text(0.0, 1.0, text, va="top", ha="left", color=TEXT_COLOR,
            fontsize=8, linespacing=1.5, transform=ax.transAxes,
            fontfamily="monospace")

    return fig


def _page_disclaimer() -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    _apply_page_style(fig)
    _header_bar(fig, "Disclaimer & Important Disclosures", "")
    _footer(fig, "Page 15")

    ax = fig.add_axes([0.07, 0.06, 0.86, 0.84])
    ax.axis("off")

    disc_text = """
IMPORTANT DISCLAIMER
════════════════════════════════════════════════════════════════════════════════

This research report ("Report") has been prepared solely for research, educational, and
informational purposes. It does not constitute investment advice, a recommendation, a
solicitation, or an offer to buy or sell any security, financial instrument, or investment
strategy.

HYPOTHETICAL PERFORMANCE DISCLOSURE
────────────────────────────────────
All performance results presented in this Report are hypothetical and based on simulated
backtesting. Hypothetical performance results have inherent limitations. Unlike actual
performance records, simulated results do not represent actual trading, and may not reflect
the impact of material economic and market factors that would have affected actual trading.
No representation is made that any account will or is likely to achieve results similar to
those shown.

RISKS AND LIMITATIONS
─────────────────────
1. Look-ahead bias: While precautions have been taken, backtests may inadvertently use
   information unavailable at the time of trading.
2. Survivorship bias: Only SPY options with available data are included; illiquid or
   delisted contracts are excluded.
3. Market impact: The model assumes all trades execute at the synthesized mid-price with
   a fixed 25 bps spread. Actual market impact for larger positions may be significantly
   higher, particularly for less liquid option contracts.
4. Overfitting: Strategy parameters were not extensively tuned, but the backtest period
   may not be representative of future market conditions.
5. Regime changes: The model was developed and tested in a specific macroeconomic regime
   (2022–2024) characterized by elevated interest rates and a generally rising equity market.
   Performance in different regimes (e.g., low-volatility, bear markets) may differ materially.
6. Transaction costs: Brokerage commissions, exchange fees, and financing costs for short
   option positions are not modeled.

FORWARD-LOOKING STATEMENTS
───────────────────────────
Any statements about future performance or market conditions are forward-looking and involve
significant uncertainty. Actual results may differ materially from those projected.

NOT INVESTMENT ADVICE
──────────────────────
Nothing in this Report should be construed as investment, legal, tax, or accounting advice.
Recipients should consult with qualified advisors before making any investment decisions.

This Report is provided "as is" with no warranties of any kind, express or implied.
    """

    ax.text(0.0, 1.0, disc_text, va="top", ha="left", color=TEXT_COLOR,
            fontsize=8.5, linespacing=1.6, transform=ax.transAxes)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ReportBuilder — assembles and saves the full PDF
# ─────────────────────────────────────────────────────────────────────────────

class ReportBuilder:
    """
    Builds an institutional-quality multi-page PDF research report.

    Usage:
        rb = ReportBuilder()
        rb.build(equity_dict, trades_dict, spy_prices, metrics, convergence_results)
        rb.save_pdf("outputs/reports/crr_surface_report.pdf")
    """

    def __init__(self):
        self.figures: List[plt.Figure] = []

    def build(
        self,
        equity_dict: Dict[str, pd.DataFrame],
        spy_prices: pd.DataFrame,
        metrics: Dict,
        convergence_results: Optional[Dict] = None,
        report_date: Optional[str] = None,
    ):
        """
        Build all report pages in order.

        Args:
            equity_dict:         {label: equity_df} for each bucket
            spy_prices:          SPY daily OHLC DataFrame with DatetimeIndex
            metrics:             {label: full_performance_summary()} for each bucket
            convergence_results: output of run_full_convergence_suite() or None
            report_date:         date string for cover page (defaults to today)
        """
        if report_date is None:
            report_date = str(_date.today())

        # Derive date range string from equity data
        all_dates = []
        for eq in equity_dict.values():
            if not eq.empty:
                col = eq["date"] if "date" in eq.columns else eq.index
                all_dates.extend(pd.to_datetime(col).tolist())
        if all_dates:
            start_yr = min(all_dates).year
            end_yr = max(all_dates).year
            date_range_str = f"{start_yr}-{end_yr}" if start_yr != end_yr else str(start_yr)
        else:
            date_range_str = "Multi-Year"

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        })

        print("  Building page 1: Cover ...")
        self.figures.append(_page_cover(report_date, date_range_str))

        print("  Building page 2: Executive Summary ...")
        self.figures.append(_page_exec_summary(metrics))

        print("  Building page 3: Equity Curve ...")
        self.figures.append(_page_equity_curve(equity_dict, spy_prices))

        print("  Building page 4: Drawdown Analysis ...")
        self.figures.append(_page_drawdown(equity_dict))

        print("  Building page 5: Rolling Metrics ...")
        self.figures.append(_page_rolling_metrics(equity_dict))

        print("  Building page 6: Monthly P&L ...")
        self.figures.append(_page_monthly_pnl(metrics))

        print("  Building page 7: Yearly Summary ...")
        self.figures.append(_page_yearly_summary(metrics))

        print("  Building page 8: Benchmark Comparison ...")
        self.figures.append(_page_benchmark_comparison(metrics, spy_prices, equity_dict))

        print("  Building page 9: Alpha Significance ...")
        self.figures.append(_page_alpha_significance(metrics))

        print("  Building page 10: Trade Statistics ...")
        self.figures.append(_page_trade_statistics(metrics))

        print("  Building page 11: Bucket Comparison ...")
        self.figures.append(_page_bucket_comparison(equity_dict, metrics))

        print("  Building page 12: Regime Analysis ...")
        self.figures.append(_page_regime_analysis(metrics, spy_prices, equity_dict))

        print("  Building page 13: CRR Convergence ...")
        self.figures.append(_page_convergence(convergence_results))

        print("  Building page 14: Methodology ...")
        self.figures.append(_page_methodology())

        print("  Building page 15: Disclaimer ...")
        self.figures.append(_page_disclaimer())

    def save_pdf(self, path: str | Path):
        """Save all figures to a multi-page PDF."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(str(path)) as pdf:
            for i, fig in enumerate(self.figures):
                pdf.savefig(fig, dpi=DPI, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved page {i+1}/{len(self.figures)}")

        print(f"\nReport saved: {path}")
        print(f"Total pages: {len(self.figures)}")

    def save_figures_png(self, figures_dir: str | Path):
        """
        Save each report page as an individual high-DPI PNG for LaTeX inclusion.
        Files are named: fig01_cover.png, fig02_exec_summary.png, etc.
        Call this BEFORE save_pdf (save_pdf closes the figures).
        """
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        page_names = [
            "cover", "exec_summary", "equity_curve", "drawdown",
            "rolling_metrics", "monthly_pnl", "yearly_summary",
            "benchmark", "alpha_significance", "trade_statistics",
            "bucket_comparison", "regime_analysis", "convergence",
            "methodology", "disclaimer",
        ]

        for i, fig in enumerate(self.figures):
            name = page_names[i] if i < len(page_names) else f"page{i+1:02d}"
            out_path = figures_dir / f"fig{i+1:02d}_{name}.png"
            fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight",
                        facecolor="white")
            print(f"  Saved figure: {out_path.name}")

        print(f"\nAll figures saved to: {figures_dir}")
