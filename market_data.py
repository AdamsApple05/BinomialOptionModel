import pandas as pd
from polygon import RESTClient


class MarketData:
    """Interface for Polygon.io historical data retrieval."""

    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)

    def get_spot_at_date(self, symbol: str, date: str) -> float:
        """Finds the first valid closing price on or after the requested date."""
        current_date = pd.to_datetime(date)
        for _ in range(10):
            try:
                aggs = self.client.get_aggs(symbol, 1, "day", current_date.strftime(
                    '%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
                if aggs:
                    return aggs[0].close
            except:
                pass
            current_date += pd.Timedelta(days=1)
        raise ValueError(f"Spot data missing for {symbol} near {date}")

    def find_optimal_ticker(self, symbol, start_date, expiry_days, option_type):
        """Identifies the best-fit 180D 10% OTM contract as of the start date."""
        spot = self.get_spot_at_date(symbol, start_date)
        target_strike = spot * 1.10 if option_type == "call" else spot * 0.90
        target_dt = pd.to_datetime(start_date) + pd.Timedelta(days=expiry_days)

        try:
            contracts = list(self.client.list_options_contracts(
                underlying_ticker=symbol, limit=1000, as_of=start_date))
            best = min(contracts, key=lambda c: abs((pd.to_datetime(
                c.expiration_date) - target_dt).days) * 2 + abs(c.strike_price - target_strike))
            return best.ticker
        except Exception as e:
            raise RuntimeError(
                f"Polygon Discovery Error for {symbol}: {str(e)}")
