from __future__ import annotations

from dataclasses import dataclass

from core.alpaca_rest import AlpacaAccount, AlpacaOrder, AlpacaPosition, AlpacaRestClient
from core.portfolio_manager import PortfolioManager


@dataclass(frozen=True)
class BrokerState:
    account: AlpacaAccount
    positions: list[AlpacaPosition]
    open_orders: list[AlpacaOrder]


class BrokerReconciler:
    def __init__(self, *, alpaca: AlpacaRestClient, portfolio: PortfolioManager):
        self.alpaca = alpaca
        self.portfolio = portfolio

    def reconcile(self) -> BrokerState:
        # Read-only reconciliation.
        acct = self.alpaca.get_account()
        positions = self.alpaca.get_positions()
        open_orders = self.alpaca.get_open_orders()

        # Sync portfolio holdings to broker truth.
        self.portfolio.sync_from_broker_positions(positions)

        return BrokerState(account=acct, positions=positions, open_orders=open_orders)
