import asyncio

from core.async_stream_engine import AlpacaStream
from core.trading_halt import clear_halt_for_tests, halt_reason, is_halted


def test_ws_connect_checks_kill_switch(monkeypatch):
    monkeypatch.setenv("KILL_SWITCH", "1")
    clear_halt_for_tests()

    stream = AlpacaStream(["AAPL"], max_failures=2)
    try:
        asyncio.run(stream.connect())
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("expected SystemExit")


def test_ws_reconnect_failures_trigger_halt(monkeypatch):
    monkeypatch.setenv("KILL_SWITCH", "0")
    clear_halt_for_tests()

    async def no_sleep(_s: float) -> None:
        return None

    def boom(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr("core.async_stream_engine.asyncio.sleep", no_sleep)
    monkeypatch.setattr("core.async_stream_engine.websockets.connect", boom)

    stream = AlpacaStream(["AAPL"], max_failures=2)
    asyncio.run(stream.connect())

    assert is_halted() is True
    assert "ws_reconnect_failures" in (halt_reason() or "")
