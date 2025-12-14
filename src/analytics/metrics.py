from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def _as_series(values: Iterable[float] | pd.Series) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(list(values), dtype=float)

    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    return series.astype(float)


def annualization_factor(periods_per_year: int) -> float:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return float(periods_per_year)


def volatility(returns: Iterable[float] | pd.Series, *, periods_per_year: int = 252) -> float:
    r = _as_series(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(annualization_factor(periods_per_year)))


def sharpe_ratio(
    returns: Iterable[float] | pd.Series,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    r = _as_series(returns)
    if len(r) < 2:
        return float("nan")

    rf_per_period = risk_free_rate / annualization_factor(periods_per_year)
    excess = r - rf_per_period
    denom = excess.std(ddof=1)
    if denom == 0:
        return float("nan")
    return float(excess.mean() / denom * np.sqrt(annualization_factor(periods_per_year)))


def downside_deviation(
    returns: Iterable[float] | pd.Series,
    *,
    periods_per_year: int = 252,
    minimum_acceptable_return: float = 0.0,
) -> float:
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")

    mar_per_period = minimum_acceptable_return / annualization_factor(periods_per_year)
    downside = np.minimum(r - mar_per_period, 0.0)
    # Use population mean of squared downside; annualize via sqrt(T)
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(annualization_factor(periods_per_year)))


def sortino_ratio(
    returns: Iterable[float] | pd.Series,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    minimum_acceptable_return: float = 0.0,
) -> float:
    r = _as_series(returns)
    if len(r) < 2:
        return float("nan")

    rf_per_period = risk_free_rate / annualization_factor(periods_per_year)
    excess = r - rf_per_period
    dd = downside_deviation(
        r,
        periods_per_year=periods_per_year,
        minimum_acceptable_return=minimum_acceptable_return,
    )
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float(excess.mean() * annualization_factor(periods_per_year) / dd)


def max_drawdown(equity: Iterable[float] | pd.Series) -> float:
    curve = _as_series(equity)
    if len(curve) == 0:
        return float("nan")
    peak = curve.cummax()
    dd = (peak - curve) / peak
    return float(dd.max())


def cagr(equity: Iterable[float] | pd.Series, *, periods_per_year: int = 252) -> float:
    curve = _as_series(equity)
    if len(curve) < 2:
        return float("nan")

    start = float(curve.iloc[0])
    end = float(curve.iloc[-1])
    if start <= 0 or end <= 0:
        return float("nan")

    years = (len(curve) - 1) / annualization_factor(periods_per_year)
    if years <= 0:
        return float("nan")

    return float((end / start) ** (1.0 / years) - 1.0)


def calmar_ratio(equity: Iterable[float] | pd.Series, *, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(cagr(equity, periods_per_year=periods_per_year) / mdd)


def hit_rate(returns: Iterable[float] | pd.Series) -> float:
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


@dataclass(frozen=True)
class WinLossStats:
    avg_win: float
    avg_loss: float


def avg_win_loss(returns: Iterable[float] | pd.Series) -> WinLossStats:
    r = _as_series(returns)
    wins = r[r > 0]
    losses = r[r < 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    return WinLossStats(avg_win=avg_win, avg_loss=avg_loss)


def profit_factor(returns: Iterable[float] | pd.Series) -> float:
    r = _as_series(returns)
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


@dataclass(frozen=True)
class TailRisk:
    var: float
    es: float


def historical_var_es(returns: Iterable[float] | pd.Series, *, level: float) -> TailRisk:
    if not (0.0 < level < 1.0):
        raise ValueError("level must be between 0 and 1")

    r = _as_series(returns)
    if len(r) == 0:
        return TailRisk(var=float("nan"), es=float("nan"))

    losses = -r
    var = float(np.quantile(losses.to_numpy(), level))
    tail = losses[losses >= var]
    es = float(tail.mean()) if len(tail) else var
    return TailRisk(var=var, es=es)


def gross_exposure(weights: pd.DataFrame) -> pd.Series:
    """Gross exposure per timestamp: sum(|w_i|)."""
    return weights.abs().sum(axis=1)


def net_exposure(weights: pd.DataFrame) -> pd.Series:
    """Net exposure per timestamp: sum(w_i)."""
    return weights.sum(axis=1)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover per timestamp: 0.5 * sum(|w_t - w_{t-1}|)."""
    if len(weights.index) == 0:
        return pd.Series(dtype=float)

    delta = weights.diff().abs()
    trn = 0.5 * delta.sum(axis=1)
    trn.iloc[0] = 0.0
    return trn


def count_missing_prices(prices: pd.DataFrame) -> Mapping[str, int]:
    """Count missing prices per symbol column."""
    return {str(col): int(prices[col].isna().sum()) for col in prices.columns}
