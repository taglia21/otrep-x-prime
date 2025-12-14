from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from core.safety import SafetyError


@dataclass(frozen=True)
class HaltInfo:
    ts: float
    reason: str


_lock = threading.Lock()
_halt: HaltInfo | None = None


def halt_trading(*, reason: str) -> None:
    """Flip the process into a HALT_TRADING state.

    This is a safety mechanism: it must never print secrets or attempt orders.
    The first halt reason wins to preserve root-cause signal.
    """
    global _halt
    r = str(reason).strip() or "unspecified"
    with _lock:
        if _halt is None:
            _halt = HaltInfo(ts=time.time(), reason=r)


def clear_halt_for_tests() -> None:
    global _halt
    with _lock:
        _halt = None


def is_halted() -> bool:
    with _lock:
        return _halt is not None


def halt_reason() -> str | None:
    with _lock:
        return None if _halt is None else _halt.reason


def guard_not_halted() -> None:
    """Raise if trading has been halted.

    Intended to be called in any order-submitting path.
    """
    with _lock:
        if _halt is None:
            return
        raise SafetyError(f"Trading halted: {_halt.reason}")
