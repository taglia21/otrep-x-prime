"""
momentum_strategy.py
Simple price-momentum based strategy.
"""

class MomentumStrategy:
    def __init__(self):
        self.feed = None
        self.executor = None

    def initialize(self, feed, executor):
        self.feed = feed
        self.executor = executor

    def generate_signal(self, price):
        if price > 180:
            return -1  # Sell signal
        elif price < 160:
            return 1  # Buy signal
        return 0
