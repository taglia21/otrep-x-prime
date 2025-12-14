from __future__ import annotations

import os
from dataclasses import dataclass


_TRUE = {"1", "true", "yes", "on", "y", "t"}


def env_flag(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default)
    return str(raw).strip().lower() in _TRUE


def assert_not_killed() -> None:
    if env_flag("KILL_SWITCH", default="0"):
        raise SystemExit(2)


@dataclass(frozen=True)
class SafetyError(RuntimeError):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


def orders_allowed(*, require_ack: bool = False) -> bool:
    # Hard stop takes precedence.
    if env_flag("NO_ORDERS", default="1"):
        return False

    if not env_flag("ALLOW_ORDERS", default="0"):
        return False

    if require_ack and not env_flag("I_UNDERSTAND_LIVE_TRADING", default="0"):
        return False

    return True


def guard_order_submission(*, require_ack: bool = False) -> None:
    if orders_allowed(require_ack=require_ack):
        return

    parts: list[str] = []

    if env_flag("KILL_SWITCH", default="0"):
        parts.append("KILL_SWITCH=1 is set (immediate stop)")

    if env_flag("NO_ORDERS", default="1"):
        parts.append("NO_ORDERS=1 blocks all order submissions")

    if not env_flag("ALLOW_ORDERS", default="0"):
        parts.append("ALLOW_ORDERS=1 is required to enable order submissions")

    if require_ack and not env_flag("I_UNDERSTAND_LIVE_TRADING", default="0"):
        parts.append("I_UNDERSTAND_LIVE_TRADING=1 is required for live loops")

    guidance = (
        "Order submission blocked by safety policy. "
        "To enable intentionally (paper recommended), set: "
        "ALLOW_ORDERS=1 and NO_ORDERS=0. "
        "For live loops you may also need I_UNDERSTAND_LIVE_TRADING=1."
    )

    detail = "; ".join(parts) if parts else "orders_allowed() returned False"
    raise SafetyError(f"{guidance} ({detail})")
