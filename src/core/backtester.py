import pandas as pd
import time

class Backtester:
    """
    Simple backtesting engine that runs a strategy
    over historical price data stored in CSV format.
    """
    def __init__(self, strategy_class, data_path, logger):
        self.strategy_class = strategy_class
        self.data_path = data_path
        self.logger = logger

    def run(self):
        df = pd.read_csv(self.data_path)
        self.logger.info(f"Loaded historical data from {self.data_path} — {len(df)} rows.")
        total_trades = 0

        for _, row in df.iterrows():
            price = row['price']
            strategy = self.strategy_class(MockDataFeed(price), self.logger)
            strategy.run()
            total_trades += 1
            time.sleep(0.05)

        self.logger.info(f"Backtest complete — total simulated trades: {total_trades}")

class MockDataFeed:
    """Mocks live price feed for backtesting."""
    def __init__(self, price):
        self._price = price
    def get_price(self, _):
        return self._price
