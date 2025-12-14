from unittest.mock import Mock

import pytest

from src.core.order_journal import OrderJournal
from src.core.order_manager import OrderManager, deterministic_client_order_id


class FakeAlpaca:
    def __init__(self):
        self.submit_calls = 0

    def get_open_orders(self):
        return []

    def submit_order(self, **kwargs):
        self.submit_calls += 1
        order = Mock()
        order.id = "abc"
        order.client_order_id = kwargs.get("client_order_id")
        order.symbol = kwargs["symbol"]
        order.side = kwargs["side"]
        order.qty = float(kwargs["qty"])
        order.filled_qty = 0.0
        order.status = "accepted"
        return order, 0.01, 200

    def get_order(self, order_id: str):
        raise AssertionError("not used")


def test_order_manager_dedupes_same_client_order_id(tmp_path):
    alpaca = FakeAlpaca()
    journal = OrderJournal(tmp_path / "orders.sqlite")
    om = OrderManager(alpaca=alpaca, journal=journal)

    # Freeze time via now_ts to make client_order_id deterministic.
    d1 = om.submit_order(strategy="s", symbol="AAPL", qty=1, side="buy", now_ts=1000.0)
    d2 = om.submit_order(strategy="s", symbol="AAPL", qty=1, side="buy", now_ts=1000.0)

    assert d1.client_order_id == d2.client_order_id
    assert d1.submitted is True
    assert d2.submitted is False
    assert alpaca.submit_calls == 1


def test_client_order_id_changes_across_buckets():
    a = deterministic_client_order_id(
        strategy="s", symbol="AAPL", side="buy", qty=1, order_type="market", tif="day", now_ts=60.0, bucket_seconds=60
    )
    b = deterministic_client_order_id(
        strategy="s", symbol="AAPL", side="buy", qty=1, order_type="market", tif="day", now_ts=61.0, bucket_seconds=60
    )
    assert a != b
