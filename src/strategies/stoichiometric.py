from src.core_logger import logger
class StoichiometricStrategy:
    def __init__(self, data_feed=None, logger=logger):
        self.data_feed = data_feed
        self.logger = logger
    def generate_signal(self, row):
        price = row["close"]
        return 1.0 if price < self._mean_price() else -1.0
    def _mean_price(self):
        df = self.data_feed.df
        return float(df["close"].mean())
