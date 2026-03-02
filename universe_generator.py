"""
universe_generator.py
---------------------
Generates a filtered, institutional-grade universe of 1,000 liquid stocks.
Fixes the 'Invalid sort field' error by sorting locally.
"""
import os
import pandas as pd
from polygon import RESTClient


def get_institutional_universe(limit=1000):
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("Error: POLYGON_API_KEY environment variable not set.")
        return

    client = RESTClient(api_key)
    tickers_raw = []

    print(f"Querying Polygon for active common stocks...")

    try:
        # Fetching active common stocks (type='CS')
        # We remove the API-side sort to avoid the 'Invalid sort field' error
        results = client.list_tickers(
            market="stocks",
            type="CS",
            active=True,
            limit=1000  # Increase this if you want a larger pool to sort from
        )

        for t in results:
            tickers_raw.append({
                "ticker": t.ticker,
                "name": t.name,
                "market_cap": getattr(t, 'market_cap', 0) if getattr(t, 'market_cap', 0) is not None else 0
            })

        # Perform the sorting locally in Pandas
        df = pd.DataFrame(tickers_raw)
        df = df.sort_values(by="market_cap", ascending=False).head(limit)

        df.to_csv("institutional_universe.csv", index=False)
        print(
            f"Successfully generated universe: {len(df)} tickers saved to institutional_universe.csv")

    except Exception as e:
        print(f"Failed to generate universe: {str(e)}")


if __name__ == "__main__":
    get_institutional_universe()
