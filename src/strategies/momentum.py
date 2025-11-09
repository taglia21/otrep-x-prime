class MomentumStrategy:
    def __init__(self, data_feed=None, logger=None):
        self.data_feed = data_feed
        self.logger = logger
    def generate_signal(self, row):
        price = row["close"]
        median = float(self.data_feed.df["close"].median())
        return 1.0 if price > median else -1.0
