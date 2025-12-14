from core.alpaca_rest import AlpacaAccount, AlpacaOrder, AlpacaPosition
from core.portfolio_manager import PortfolioManager
from core.reconciler import BrokerReconciler
from core.trading_halt import clear_halt_for_tests, is_halted


class FakeAlpaca:
    def __init__(self, *, qty: float):
        self.qty = float(qty)

    def get_account(self):
        return AlpacaAccount(equity=100.0, cash=100.0)

    def get_positions(self):
        return [AlpacaPosition(symbol="AAPL", qty=self.qty, avg_entry_price=None, market_value=None)]

    def get_open_orders(self):
        return [AlpacaOrder(id="1", client_order_id=None, symbol="AAPL", side="buy", qty=1.0, filled_qty=0.0, status="open")]


def test_reconcile_halts_on_position_mismatch_after_baseline():
    clear_halt_for_tests()

    portfolio = PortfolioManager()
    alpaca = FakeAlpaca(qty=1.0)
    r = BrokerReconciler(alpaca=alpaca, portfolio=portfolio, position_mismatch_tolerance=0.0)

    # First call establishes baseline and syncs portfolio.
    r.reconcile()
    assert is_halted() is False

    # Local drift appears (e.g. due to a bug/manual intervention).
    portfolio.positions["AAPL"]["qty"] = 2.0

    # Second reconcile detects mismatch and halts.
    r.reconcile()
    assert is_halted() is True
