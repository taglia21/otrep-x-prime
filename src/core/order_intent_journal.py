from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.jsonl_journal import JsonlJournal


@dataclass(frozen=True)
class OrderIntent:
    client_order_id: str
    strategy: str
    symbol: str
    side: str
    qty: float
    order_type: str
    tif: str
    ts: float


class OrderIntentJournal:
    """JSONL journal for order intents.

    This is separate from the SQLite `OrderJournal` which stores order lifecycle.
    The intent journal is append-only and is written *before* submission.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._journal = JsonlJournal(self.path)

    def append_intent(self, *, intent: OrderIntent, extra: dict[str, Any] | None = None) -> None:
        event: dict[str, Any] = {
            "ts": float(intent.ts),
            "event": "order_intent",
            "client_order_id": intent.client_order_id,
            "strategy": intent.strategy,
            "symbol": intent.symbol,
            "side": intent.side,
            "qty": float(intent.qty),
            "order_type": intent.order_type,
            "tif": intent.tif,
        }
        if extra:
            # Ensure JSON-serializable by round-tripping through dumps.
            json.dumps(extra)
            event.update(extra)
        self._journal.append(event)

    def replay(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        return out


class OrderIntentStore:
    """In-memory view of intents loaded from a JSONL journal."""

    def __init__(self, *, journal: OrderIntentJournal):
        self.journal = journal
        self._by_client_order_id: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        for e in self.journal.replay():
            coid = e.get("client_order_id")
            if isinstance(coid, str) and coid:
                self._by_client_order_id[coid] = e
        self._loaded = True

    def has(self, client_order_id: str) -> bool:
        self.load()
        return client_order_id in self._by_client_order_id

    def register(self, *, intent: OrderIntent, extra: dict[str, Any] | None = None) -> None:
        self.load()
        d: dict[str, Any] = {
            "ts": float(intent.ts),
            "event": "order_intent",
            "client_order_id": intent.client_order_id,
            "strategy": intent.strategy,
            "symbol": intent.symbol,
            "side": intent.side,
            "qty": float(intent.qty),
            "order_type": intent.order_type,
            "tif": intent.tif,
        }
        if extra:
            d.update(extra)
        self._by_client_order_id[intent.client_order_id] = d
