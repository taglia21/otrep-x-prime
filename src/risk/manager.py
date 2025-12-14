from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reasons: list[str]


@dataclass(frozen=True)
class RiskLimits:
    max_gross_leverage: float = 1.0
    max_position_pct: float = 1.0
    max_notional_per_symbol: float = 50_000.0
    max_turnover_per_step: float = 1.0


class RiskManager:
    """Minimal, real risk manager for pre-trade checks.

    Intended for both sim/backtest and live.
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def check_rebalance(
        self,
        *,
        equity: float,
        current_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
    ) -> RiskDecision:
        reasons: list[str] = []

        if equity <= 0:
            return RiskDecision(False, ["equity<=0"])

        gross = sum(abs(float(w)) for w in target_weights.values())
        if gross > float(self.limits.max_gross_leverage) + 1e-12:
            reasons.append(f"gross_leverage {gross:.4f} > {self.limits.max_gross_leverage}")

        for sym, w in target_weights.items():
            if abs(float(w)) > float(self.limits.max_position_pct) + 1e-12:
                reasons.append(f"position_pct {sym} {float(w):.4f} exceeds {self.limits.max_position_pct}")

            notional = abs(float(w) * float(equity))
            if notional > float(self.limits.max_notional_per_symbol) + 1e-9:
                reasons.append(
                    f"notional {sym} {notional:.2f} exceeds {self.limits.max_notional_per_symbol:.2f}"
                )

        # One-way turnover: 0.5 * sum(|w_t - w_{t-1}|)
        syms = set(current_weights.keys()) | set(target_weights.keys())
        turnover = 0.5 * sum(abs(float(target_weights.get(s, 0.0)) - float(current_weights.get(s, 0.0))) for s in syms)
        if turnover > float(self.limits.max_turnover_per_step) + 1e-12:
            reasons.append(f"turnover {turnover:.4f} > {self.limits.max_turnover_per_step}")

        return RiskDecision(allowed=(len(reasons) == 0), reasons=reasons)

    def check_order(
        self,
        *,
        equity: float,
        symbol: str,
        qty: float,
        price: float,
    ) -> RiskDecision:
        reasons: list[str] = []

        if equity <= 0:
            reasons.append("equity<=0")
            return RiskDecision(False, reasons)

        notional = abs(float(qty) * float(price))
        if notional > float(self.limits.max_notional_per_symbol) + 1e-9:
            reasons.append(
                f"notional {symbol} {notional:.2f} exceeds {self.limits.max_notional_per_symbol:.2f}"
            )

        return RiskDecision(allowed=(len(reasons) == 0), reasons=reasons)
