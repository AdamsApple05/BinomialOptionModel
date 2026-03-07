from dataclasses import dataclass
import os


@dataclass(frozen=True)
class GlobalConfig:
    api_key: str = os.getenv("POLYGON_API_KEY", "YOUR_API_KEY_HERE")
    underlying_symbol: str = "SPY"

    # Pricing assumptions
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.012
    crr_steps: int = 80

    # Aggregate data assumptions
    assumed_spread_bps: float = 25.0
    trading_days_per_year: int = 252

    # Backtest dates
    start_date: str = "2024-01-01"
    end_date: str = "2024-06-30"
