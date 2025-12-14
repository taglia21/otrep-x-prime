import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.types import BacktestConfig


class EqualWeightStrategy:
    def __init__(self, weights: dict[str, float]):
        self.weights = dict(weights)

    def target_weights(self, *, t: pd.Timestamp, history: pd.DataFrame) -> dict[str, float]:
        return dict(self.weights)


def test_backtest_engine_equal_weight_no_costs_golden():
    prices = pd.read_csv("data/sample_prices.csv")

    cfg = BacktestConfig(starting_cash=10_000.0)
    engine = BacktestEngine(cfg=cfg)

    strat = EqualWeightStrategy({"AAPL": 0.5, "MSFT": 0.5})
    result = engine.run(prices=prices, strategy=strat)

    equity = result.equity_curve["equity"].round(6)
    # Expected equity values (close-to-close fills with 1-bar delay, integer shares, zero costs)
    expected = pd.Series(
        [
            10000.0,  # 2025-01-02: before first fills
            10000.0,  # 2025-01-03: after first fills
            10225.0,  # 2025-01-04
            10764.0,  # 2025-01-05
        ],
        index=pd.to_datetime(
            ["2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"], utc=True
        ),
        dtype=float,
    )

    assert equity.index.equals(expected.index)
    assert equity.tolist() == expected.tolist()

    assert result.diagnostics["missing_fill_prices"] == 0
