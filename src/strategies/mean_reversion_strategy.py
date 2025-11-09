"""
mean_reversion_strategy.py
Reverts toward historical average.
"""

class MeanReversionStrategy:
    def __init__(self):
        self.feed = None
        self.executor = None

    def initialize(self, feed, executor):
        self.feed = feed
        self.executor = executor

    def generate_signal(self, price):
        mean_price = 175
        if price < mean_price * 0.95:
            return 1  # Buy undervalued
        elif price > mean_price * 1.05:
            return -1  # Sell overvalued
        return 0
