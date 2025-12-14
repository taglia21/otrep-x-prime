from src.risk.manager import RiskLimits, RiskManager


def test_risk_manager_denies_gross_leverage():
    rm = RiskManager(RiskLimits(max_gross_leverage=1.0))
    d = rm.check_rebalance(
        equity=100_000.0,
        current_weights={"AAPL": 0.0},
        target_weights={"AAPL": 1.0, "MSFT": 1.0},
    )
    assert d.allowed is False
    assert any("gross_leverage" in r for r in d.reasons)


def test_risk_manager_denies_turnover():
    rm = RiskManager(RiskLimits(max_turnover_per_step=0.10))
    d = rm.check_rebalance(
        equity=100_000.0,
        current_weights={"AAPL": 0.0, "MSFT": 0.0},
        target_weights={"AAPL": 0.5, "MSFT": 0.5},
    )
    assert d.allowed is False
    assert any("turnover" in r for r in d.reasons)


def test_risk_manager_denies_notional():
    rm = RiskManager(RiskLimits(max_notional_per_symbol=1_000.0))
    d = rm.check_rebalance(
        equity=100_000.0,
        current_weights={"AAPL": 0.0},
        target_weights={"AAPL": 0.05},
    )
    assert d.allowed is False
    assert any("notional" in r for r in d.reasons)
