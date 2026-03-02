import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polygon import RESTClient
import os


def generate_report():
    """Generates Risk-Adjusted Sharpe, Beta, and VIX Correlation."""
    if not os.path.exists("backtest_results/master_portfolio_equity.csv"):
        return

    df = pd.read_csv(
        "backtest_results/master_portfolio_equity.csv", index_col=0)
    df.index = pd.to_datetime(df.index).tz_localize(
        None).astype('datetime64[ns]')
    portfolio_equity = df.sum(axis=1)
    port_rets = portfolio_equity.pct_change().dropna()

    client = RESTClient(os.getenv("POLYGON_API_KEY"))
    start, end = df.index.min().strftime(
        '%Y-%m-%d'), df.index.max().strftime('%Y-%m-%d')

    spy = pd.DataFrame([{"date": pd.to_datetime(b.timestamp, unit='ms'), "spy": b.close}
                       for b in client.get_aggs("SPY", 1, "day", start, end)])
    spy["date"] = spy["date"].dt.tz_localize(None).astype('datetime64[ns]')

    vxx = pd.DataFrame([{"date": pd.to_datetime(b.timestamp, unit='ms'), "vxx": b.close}
                       for b in client.get_aggs("VXX", 1, "day", start, end)])
    vxx["date"] = vxx["date"].dt.tz_localize(None).astype('datetime64[ns]')

    # Multi-source merge_asof for perfect alignment
    final = pd.merge_asof(port_rets.to_frame('p_rets'), spy.set_index(
        'date').pct_change(), left_index=True, right_index=True)
    final = pd.merge_asof(final, vxx.set_index(
        'date').pct_change(), left_index=True, right_index=True).dropna()

    print("-" * 30)
    print(
        f"Sharpe Ratio:    {port_rets.mean()/port_rets.std() * np.sqrt(252):.2f}")
    print(
        f"Portfolio Beta:  {np.cov(final['p_rets'], final['spy'])[0][1]/np.var(final['spy']):.3f}")
    print(f"VXX Correlation: {final['p_rets'].corr(final['vxx']):.2%}")
    print("-" * 30)

    portfolio_equity.plot(title="Institutional Equity Curve", color='cyan')
    plt.show()


if __name__ == "__main__":
    generate_report()
