from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .types import BacktestConfig, Strategy
from src.risk.manager import RiskLimits, RiskManager


@dataclass(frozen=True)
class TradeFill:
    t_fill: pd.Timestamp
    symbol: str
    shares: int
    price: float
    notional: float
    costs: float


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame  # index=t, cols: equity,cash
    positions: pd.DataFrame  # index=t, cols per symbol shares
    fills: pd.DataFrame  # long-form fills
    diagnostics: dict[str, Any]


def _to_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize inputs to a wide price panel indexed by timestamp with symbol columns."""
    if {"timestamp", "symbol", "close"}.issubset(prices.columns):
        df = prices.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        panel = df.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last").sort_index()
        return panel

    if {"date", "symbol", "close"}.issubset(prices.columns):
        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        panel = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
        panel.index.name = "timestamp"
        return panel

    if isinstance(prices.index, pd.DatetimeIndex):
        return prices.sort_index()

    raise ValueError("Unsupported price dataframe format")


def _truthy_weights(targets: dict[str, float]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for k, v in targets.items():
        if v is None:
            continue
        cleaned[str(k)] = float(v)
    return cleaned


def _compute_costs(*, shares: int, price: float, cfg: BacktestConfig) -> float:
    notional = abs(float(shares) * float(price))
    costs = 0.0
    costs += cfg.costs.commission_fixed if shares != 0 else 0.0
    costs += cfg.costs.commission_per_share * abs(int(shares))
    costs += (cfg.costs.slippage_bps / 10_000.0) * notional
    return float(costs)


def _affordable_buy_shares(*, cash: float, price: float, cfg: BacktestConfig) -> int:
    """Compute max buy shares affordable given cash and cost model (deterministic, conservative)."""
    if price <= 0:
        return 0

    # Conservative: include fixed commission if we buy at least 1 share.
    fixed = cfg.costs.commission_fixed
    per_share = cfg.costs.commission_per_share
    slip_mult = 1.0 + (cfg.costs.slippage_bps / 10_000.0)

    budget = cash - fixed
    if budget <= 0:
        return 0

    denom = price * slip_mult + per_share
    if denom <= 0:
        return 0

    return max(int(np.floor(budget / denom)), 0)


class BacktestEngine:
    """Deterministic backtest engine with one-bar delayed fills (no look-ahead)."""

    def __init__(self, *, cfg: BacktestConfig):
        self.cfg = cfg

    def run(
        self,
        *,
        prices: pd.DataFrame,
        strategy: Strategy,
        risk_manager: RiskManager | None = None,
    ) -> BacktestResult:
        panel = _to_price_panel(prices)

        if self.cfg.fill_delay_bars < 1:
            raise ValueError("fill_delay_bars must be >= 1")

        if len(panel.index) <= self.cfg.fill_delay_bars:
            raise ValueError("Not enough bars for the configured fill delay")

        symbols = sorted([str(c) for c in panel.columns])

        cash = float(self.cfg.starting_cash)
        holdings = {s: 0 for s in symbols}

        equity_rows: list[dict[str, Any]] = []
        pos_rows: list[dict[str, Any]] = []
        fills: list[TradeFill] = []

        missing_fill_prices = 0
        risk_denied_steps = 0
        last_risk_reasons: list[str] = []

        rm = risk_manager or RiskManager(RiskLimits())

        idx = panel.index.to_numpy()
        for i in range(0, len(panel.index) - self.cfg.fill_delay_bars):
            t_decide = pd.Timestamp(idx[i])
            t_fill = pd.Timestamp(idx[i + self.cfg.fill_delay_bars])
            if t_decide is pd.NaT or t_fill is pd.NaT:
                continue

            t_decide = cast(pd.Timestamp, t_decide)
            t_fill = cast(pd.Timestamp, t_fill)

            # History includes prices through decision time (inclusive).
            history = panel.iloc[: i + 1].copy()

            # Mark-to-market at decision time.
            px_now = panel.iloc[i]
            equity_now = cash + sum(holdings[s] * float(px_now.get(s, np.nan)) for s in symbols if pd.notna(px_now.get(s, np.nan)))

            equity_rows.append({"timestamp": t_decide, "equity": float(equity_now), "cash": float(cash)})
            pos_rows.append({"timestamp": t_decide, **{s: int(holdings[s]) for s in symbols}})

            targets = _truthy_weights(strategy.target_weights(t=t_decide, history=history))

            # Risk: validate proposed rebalance in weight space before trading.
            current_weights: dict[str, float] = {}
            for s in symbols:
                p = float(px_now.get(s, np.nan))
                if np.isfinite(p) and p > 0 and equity_now != 0:
                    current_weights[s] = float(holdings[s] * p / equity_now)
                else:
                    current_weights[s] = 0.0

            decision = rm.check_rebalance(equity=equity_now, current_weights=current_weights, target_weights=targets)
            if not decision.allowed:
                risk_denied_steps += 1
                last_risk_reasons = list(decision.reasons)
                continue

            # Fill prices at t_fill.
            px_fill = panel.loc[t_fill]

            # Convert targets to desired shares using fill prices.
            desired_shares: dict[str, int] = {}
            for s in symbols:
                w = float(targets.get(s, 0.0))
                price = float(px_fill.get(s, np.nan))
                if not np.isfinite(price) or price <= 0:
                    missing_fill_prices += 1
                    desired_shares[s] = holdings[s]
                    continue

                desired_dollars = w * equity_now
                desired_shares[s] = int(np.floor(desired_dollars / price))

            # Execute trades in deterministic symbol order.
            deltas = {s: int(desired_shares[s] - holdings[s]) for s in symbols}

            # Sell-first then buy reduces path dependence from cash constraints.
            sell_symbols = [s for s in symbols if deltas[s] < 0]
            buy_symbols = [s for s in symbols if deltas[s] > 0]

            for phase_symbols in (sell_symbols, buy_symbols):
                for s in phase_symbols:
                    delta = int(deltas[s])
                    if delta == 0:
                        continue

                    price = float(px_fill.get(s, np.nan))
                    if not np.isfinite(price) or price <= 0:
                        missing_fill_prices += 1
                        continue

                    # Enforce cash constraint for buys (sell always allowed).
                    if delta > 0:
                        max_buy = _affordable_buy_shares(cash=cash, price=price, cfg=self.cfg)
                        if max_buy <= 0:
                            continue
                        delta = min(delta, max_buy)
                        if delta <= 0:
                            continue

                    costs = _compute_costs(shares=delta, price=price, cfg=self.cfg)
                    notional = float(delta * price)

                    cash -= notional
                    cash -= costs
                    holdings[s] += delta

                    fills.append(
                        TradeFill(
                            t_fill=t_fill,
                            symbol=s,
                            shares=delta,
                            price=price,
                            notional=notional,
                            costs=costs,
                        )
                    )

        # Final mark at last available timestamp
        t_last = panel.index[-1]
        px_last = panel.iloc[-1]
        equity_last = cash + sum(holdings[s] * float(px_last.get(s, np.nan)) for s in symbols if pd.notna(px_last.get(s, np.nan)))
        equity_rows.append({"timestamp": t_last, "equity": float(equity_last), "cash": float(cash)})
        pos_rows.append({"timestamp": t_last, **{s: int(holdings[s]) for s in symbols}})

        equity_df = pd.DataFrame(equity_rows).set_index("timestamp").sort_index()
        positions_df = pd.DataFrame(pos_rows).set_index("timestamp").sort_index()

        fills_df = pd.DataFrame([
            {
                "timestamp": f.t_fill,
                "symbol": f.symbol,
                "shares": f.shares,
                "price": f.price,
                "notional": f.notional,
                "costs": f.costs,
            }
            for f in fills
        ])
        if len(fills_df) > 0:
            fills_df["timestamp"] = pd.to_datetime(fills_df["timestamp"], utc=True)
            fills_df = fills_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        diagnostics = {
            "symbols": symbols,
            "fill_delay_bars": int(self.cfg.fill_delay_bars),
            "starting_cash": float(self.cfg.starting_cash),
            "missing_fill_prices": int(missing_fill_prices),
            "risk_denied_steps": int(risk_denied_steps),
            "last_risk_reasons": last_risk_reasons,
        }

        return BacktestResult(
            equity_curve=equity_df,
            positions=positions_df,
            fills=fills_df,
            diagnostics=diagnostics,
        )
