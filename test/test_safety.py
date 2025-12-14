import os

import pytest

from src.core.safety import assert_not_killed, guard_order_submission


def test_orders_blocked_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_ORDERS", raising=False)
    monkeypatch.delenv("NO_ORDERS", raising=False)

    with pytest.raises(Exception):
        guard_order_submission()


def test_orders_allowed_only_with_allow_orders(monkeypatch):
    monkeypatch.setenv("NO_ORDERS", "0")
    monkeypatch.setenv("ALLOW_ORDERS", "1")

    guard_order_submission()


def test_no_orders_hard_blocks_even_if_allow_orders(monkeypatch):
    monkeypatch.setenv("NO_ORDERS", "1")
    monkeypatch.setenv("ALLOW_ORDERS", "1")

    with pytest.raises(Exception):
        guard_order_submission()


def test_kill_switch_raises_system_exit(monkeypatch):
    monkeypatch.setenv("KILL_SWITCH", "1")

    with pytest.raises(SystemExit) as e:
        assert_not_killed()

    assert e.value.code == 2
