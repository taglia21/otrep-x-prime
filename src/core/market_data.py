from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.alpaca_rest import AlpacaRestClient


@dataclass(frozen=True)
class RecentStats:
    returns: list[float]
    realized_vol: float
    last_price: float | None


class MarketData:
    def __init__(
        self,
        *,
        alpaca: AlpacaRestClient,
        timeframe: str = "1Min",
        bars_limit: int = 200,
    ):
        self.alpaca = alpaca
        self.timeframe = timeframe
        self.bars_limit = int(bars_limit)

    def recent_returns_and_vol(self, *, symbol: str, n_returns: int = 50) -> RecentStats:
        bars = self.alpaca.get_recent_bars(symbol=symbol, timeframe=self.timeframe, limit=self.bars_limit)
        closes: list[float] = []
        for b in bars:
            c = b.get("c")
            if c is None:
                continue
            closes.append(float(c))

        if len(closes) < 3:
            return RecentStats(returns=[], realized_vol=float("nan"), last_price=(float(closes[-1]) if closes else None))

        prices = np.asarray(closes, dtype=float)
        rets = prices[1:] / prices[:-1] - 1.0
        rets = rets[-int(n_returns) :]

        vol = float(np.std(rets, ddof=1)) if len(rets) > 1 else float("nan")
        return RecentStats(
            returns=[float(x) for x in rets.tolist()],
            realized_vol=vol,
            last_price=float(prices[-1]),
        )
