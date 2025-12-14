from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    """Strategy contract: compute target weights using information through time t."""

    def target_weights(self, *, t: pd.Timestamp, history: pd.DataFrame) -> dict[str, float]:
        """Return target weights by symbol (e.g., {'AAPL': 0.5, 'MSFT': 0.5})."""


@dataclass(frozen=True)
class CostModel:
    commission_fixed: float = 0.0
    commission_per_share: float = 0.0
    slippage_bps: float = 0.0  # bps of notional


@dataclass(frozen=True)
class BacktestConfig:
    starting_cash: float = 1_000_000.0
    costs: CostModel = CostModel()
    fill_delay_bars: int = 1
    fill_field: str = "close"  # price field used for fills
