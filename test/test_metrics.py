import numpy as np
import pandas as pd
import pytest

from src.analytics.metrics import (
    TailRisk,
    WinLossStats,
    calmar_ratio,
    cagr,
    historical_var_es,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    volatility,
)


def test_vol_sharpe_sortino_deterministic():
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.0], dtype=float)

    vol = volatility(r, periods_per_year=252)
    assert np.isfinite(vol)

    sharpe = sharpe_ratio(r, periods_per_year=252, risk_free_rate=0.0)
    assert np.isfinite(sharpe)

    sortino = sortino_ratio(r, periods_per_year=252, risk_free_rate=0.0)
    assert np.isfinite(sortino)


def test_max_drawdown_cagr_calmar_known_values():
    equity = pd.Series([100.0, 110.0, 105.0, 120.0, 90.0, 95.0], dtype=float)

    mdd = max_drawdown(equity)
    assert mdd == pytest.approx(0.25)

    # 5 periods between first and last points.
    g = cagr(equity, periods_per_year=252)
    assert np.isfinite(g)

    calmar = calmar_ratio(equity, periods_per_year=252)
    assert np.isfinite(calmar)


def test_tail_risk_var_es_convention():
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.0], dtype=float)
    tr = historical_var_es(r, level=0.95)
    assert isinstance(tr, TailRisk)

    # Losses are -returns; at 95% we should be in the right tail of losses.
    assert tr.var >= 0.0
    assert tr.es >= tr.var


def test_hit_rate_and_profit_factor():
    r = pd.Series([0.1, -0.05, 0.02, -0.01, 0.0], dtype=float)
    assert hit_rate(r) == pytest.approx(2 / 5)

    pf = profit_factor(r)
    assert pf > 1.0
