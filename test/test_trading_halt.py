import pytest

from core.alpaca_rest import AlpacaRestClient
from core.safety import SafetyError
from core.trading_halt import clear_halt_for_tests, halt_trading


def test_halt_blocks_order_submission_even_if_orders_allowed(monkeypatch):
    monkeypatch.setenv("NO_ORDERS", "0")
    monkeypatch.setenv("ALLOW_ORDERS", "1")

    clear_halt_for_tests()
    halt_trading(reason="unit_test")

    client = AlpacaRestClient(paper=True)

    # If we ever try to hit the network, this test should fail.
    def boom(*args, **kwargs):
        raise AssertionError("network should not be called when halted")

    monkeypatch.setattr("core.alpaca_rest.requests.request", boom)

    with pytest.raises(SafetyError):
        client.submit_order(symbol="AAPL", qty=1, side="buy")
