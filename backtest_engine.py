from market_data import MarketData
from zscore_strategy import ZScoreMispricingStrategy


class OptionBacktester:
    def __init__(self, api_key: str, initial_capital: float = 1_000_000.0):
        self.market_data = MarketData(api_key)
        self.initial_capital = initial_capital
        self.equity_curve = []

    def run_backtest(self, symbol, start_date, end_date, expiry_days, option_type, **params):
        # 1. Discovery Phase
        option_ticker = self.market_data.find_optimal_ticker(
            symbol, start_date, expiry_days, option_type)

        # 2. Strike Parsing
        try:
            strike = float(option_ticker[-8:]) / 1000
        except:
            strike = 0.0

        # 3. Execution Phase - FIXED: Remove initial_capital from params to avoid conflict
        strategy_params = params.copy()
        if "initial_capital" in strategy_params:
            del strategy_params["initial_capital"]

        strategy = ZScoreMispricingStrategy(
            self.initial_capital, **strategy_params)
        results = strategy.run(symbol, option_ticker,
                               strike, start_date, end_date)

        self.equity_curve = strategy.equity_curve
        return results
