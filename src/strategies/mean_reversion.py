class MeanReversionStrategy:
    def __init__(self, data_feed=None, logger=None):
        self.data_feed = data_feed
        self.logger = logger
    def generate_signal(self, row):
        price = row["close"]
        avg = float(self.data_feed.df["close"].rolling(3, min_periods=1).mean().iloc[-1])
        return -1.0 if price > avg else 1.0
