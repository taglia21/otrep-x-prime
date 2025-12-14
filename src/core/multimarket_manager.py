"""
multimarket_manager.py
Handles synchronized data feeds and execution across multiple markets.
"""

import logging
import pandas as pd
from datetime import timedelta

log = logging.getLogger(__name__)


class MarketFeed:
    """Single-market feed (e.g., US, EU, Asia)"""
    def __init__(self, name, data: pd.DataFrame, tz_offset_hours: int):
        self.name = name
        self.data = data
        self.tz_offset = timedelta(hours=tz_offset_hours)
        self.pointer = 0

    def get_next(self):
        """Return the next row adjusted for time zone offset."""
        if self.pointer >= len(self.data):
            return None
        row = self.data.iloc[self.pointer].copy()
        row["datetime_local"] = row["datetime"] + self.tz_offset
        self.pointer += 1
        return row


class MultiMarketManager:
    """Synchronizes multiple regional feeds."""
    def __init__(self, feeds: dict):
        self.feeds = feeds

    def step_all(self):
        """
        Advance all feeds simultaneously by local market time.
        Returns combined frame sorted chronologically.
        """
        frames = []
        for name, feed in self.feeds.items():
            point = feed.get_next()
            if point is not None:
                point["market"] = name
                frames.append(point)
        if not frames:
            return None
        merged = pd.DataFrame(frames).sort_values("datetime_local")
        return merged
