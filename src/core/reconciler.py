from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.alpaca_rest import AlpacaAccount, AlpacaOrder, AlpacaPosition
from core.portfolio_manager import PortfolioManager
from core.trading_halt import halt_trading


@dataclass(frozen=True)
class BrokerState:
    account: AlpacaAccount
    positions: list[AlpacaPosition]
    open_orders: list[AlpacaOrder]


class AlpacaReadOnlyLike(Protocol):
    def get_account(self) -> AlpacaAccount:
        ...

    def get_positions(self) -> list[AlpacaPosition]:
        ...

    def get_open_orders(self) -> list[AlpacaOrder]:
        ...


class BrokerReconciler:
    def __init__(
        self,
        *,
        alpaca: AlpacaReadOnlyLike,
        portfolio: PortfolioManager,
        position_mismatch_tolerance: float = 0.0,
    ):
        self.alpaca = alpaca
        self.portfolio = portfolio
        self.position_mismatch_tolerance = float(position_mismatch_tolerance)
        self._have_baseline = False

    def reconcile(self) -> BrokerState:
        # Read-only reconciliation.
        acct = self.alpaca.get_account()
        positions = self.alpaca.get_positions()
        open_orders = self.alpaca.get_open_orders()

        # Detect unexpected drift *after* we have a baseline.
        if self._have_baseline:
            try:
                before = self.portfolio.holdings_snapshot()
                broker = {p.symbol: float(p.qty) for p in positions}
                symbols = set(before.keys()) | set(broker.keys())
                tol = float(self.position_mismatch_tolerance)
                for sym in symbols:
                    a = float((before.get(sym) or {}).get("qty", 0.0))
                    b = float(broker.get(sym, 0.0))
                    if abs(a - b) > tol:
                        halt_trading(reason=f"reconcile_position_mismatch symbol={sym} local={a} broker={b} tol={tol}")
                        break
            except Exception:
                # Do not fail reconciliation due to mismatch detection errors.
                pass

        # Sync portfolio holdings to broker truth.
        self.portfolio.sync_from_broker_positions(positions)
        self._have_baseline = True

        return BrokerState(account=acct, positions=positions, open_orders=open_orders)
