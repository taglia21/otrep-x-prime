import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.types import BacktestConfig
from src.risk.manager import RiskLimits, RiskManager


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


def test_backtest_engine_risk_denies_rebalance_blocks_trades():
    prices = pd.read_csv("data/sample_prices.csv")

    # Force a denial via very low turnover limit.
    rm = RiskManager(RiskLimits(max_turnover_per_step=0.01))
    engine = BacktestEngine(cfg=BacktestConfig(starting_cash=10_000.0))

    strat = EqualWeightStrategy({"AAPL": 0.5, "MSFT": 0.5})
    result = engine.run(prices=prices, strategy=strat, risk_manager=rm)

    assert result.diagnostics["risk_denied_steps"] > 0
    assert len(result.fills) == 0


def test_backtest_engine_handles_missing_fill_prices():
    """Test that missing fill prices are properly tracked and trades are skipped."""
    # Create data with a missing price at fill time
    timestamps = pd.to_datetime(
        ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"], utc=True
    )
    prices = pd.DataFrame(
        {
            "timestamp": list(timestamps) * 2,
            "symbol": ["A"] * 4 + ["B"] * 4,
            "close": [100.0, 110.0, 120.0, 130.0, 50.0, float("nan"), 60.0, 65.0],
        }
    )

    cfg = BacktestConfig(starting_cash=10_000.0, fill_delay_bars=1)
    engine = BacktestEngine(cfg=cfg)

    strat = EqualWeightStrategy({"A": 0.5, "B": 0.5})
    result = engine.run(prices=prices, strategy=strat)

    # With fill_delay_bars=1:
    # - i=0: decide at 2025-01-01, fill at 2025-01-02 (A=110, B=NaN) <- B has missing price
    # - i=1: decide at 2025-01-02, fill at 2025-01-03 (A=120, B=60) <- both OK
    # - i=2: decide at 2025-01-03, fill at 2025-01-04 (A=130, B=65) <- both OK

    # Should detect 1 missing price (B at 2025-01-02)
    assert result.diagnostics["missing_fill_prices"] == 1

    # Should have some fills (A should have filled even when B was missing)
    assert len(result.fills) > 0


def test_backtest_engine_preserves_nan_timestamps():
    """Test that timestamps with NaN prices are preserved in the price panel."""
    # Test data with a NaN row
    prices = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"], utc=True),
            "symbol": ["TEST"] * 3,
            "close": [100.0, float("nan"), 110.0],
        }
    )

    from src.backtest.engine import _to_price_panel

    panel = _to_price_panel(prices)

    # All 3 timestamps should be preserved
    assert len(panel) == 3
    assert panel.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert panel.index[1] == pd.Timestamp("2025-01-02", tz="UTC")
    assert panel.index[2] == pd.Timestamp("2025-01-03", tz="UTC")
