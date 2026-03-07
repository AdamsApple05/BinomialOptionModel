from __future__ import annotations
from typing import Dict

from strategy import CRRSurfaceDeltaHedgeStrategy, StrategyConfig
from universe import BucketSpec


def run_bucket_backtest(
    api_key: str,
    bucket: BucketSpec,
    start_date: str,
    end_date: str,
    risk_free_rate: float,
    dividend_yield: float,
    crr_steps: int,
    assumed_spread_bps: float,
    strategy_config: StrategyConfig | None = None,
) -> Dict:
    strat = CRRSurfaceDeltaHedgeStrategy(
        api_key=api_key,
        bucket=bucket,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        crr_steps=crr_steps,
        assumed_spread_bps=assumed_spread_bps,
        config=strategy_config,
    )
    return strat.run_backtest(start_date=start_date, end_date=end_date)
