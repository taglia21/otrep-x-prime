import json

from core.order_intent_journal import OrderIntentJournal
from core.order_journal import OrderJournal
from core.order_manager import OrderManager


class ExplodingAlpaca:
    def __init__(self):
        self.submit_calls = 0

    def get_open_orders(self):
        return []

    def submit_order(self, **kwargs):
        self.submit_calls += 1
        raise RuntimeError("boom")

    def get_order(self, order_id: str):
        raise AssertionError("not used")


def test_intent_is_written_before_submission_attempt(tmp_path):
    alpaca = ExplodingAlpaca()
    journal = OrderJournal(tmp_path / "orders.sqlite")
    intent_journal = OrderIntentJournal(tmp_path / "order_intents.jsonl")

    om = OrderManager(alpaca=alpaca, journal=journal, intent_journal=intent_journal)

    d = om.submit_order(strategy="s", symbol="AAPL", qty=1, side="buy", now_ts=1000.0)

    assert d.submitted is False
    assert "submit_exception" in d.reason
    assert alpaca.submit_calls == 1

    lines = (tmp_path / "order_intents.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    evt = json.loads(lines[0])
    assert evt["event"] == "order_intent"
    assert evt["symbol"] == "AAPL"
    assert evt["side"] == "buy"
    assert evt["client_order_id"] == d.client_order_id

    row = journal.get(d.client_order_id)
    assert row is not None
    assert row.status.startswith("submit_exception_")
