from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JournalOrder:
    client_order_id: str
    alpaca_order_id: str | None
    symbol: str
    side: str
    qty: float
    status: str
    filled_qty: float


class OrderJournal:
    """SQLite-backed order journal for idempotency + restart recovery.

    Stores a minimal order lifecycle keyed by client_order_id.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
              client_order_id TEXT PRIMARY KEY,
              alpaca_order_id TEXT,
              symbol TEXT NOT NULL,
              side TEXT NOT NULL,
              qty REAL NOT NULL,
              status TEXT NOT NULL,
              filled_qty REAL NOT NULL,
              created_ts REAL NOT NULL,
              updated_ts REAL NOT NULL,
              last_payload_json TEXT
            );
            """
        )
        self._conn.commit()

    def get(self, client_order_id: str) -> JournalOrder | None:
        cur = self._conn.execute(
            "SELECT client_order_id, alpaca_order_id, symbol, side, qty, status, filled_qty FROM orders WHERE client_order_id=?",
            (client_order_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return JournalOrder(
            client_order_id=str(row[0]),
            alpaca_order_id=row[1],
            symbol=str(row[2]),
            side=str(row[3]),
            qty=float(row[4]),
            status=str(row[5]),
            filled_qty=float(row[6]),
        )

    def upsert(
        self,
        *,
        client_order_id: str,
        alpaca_order_id: str | None,
        symbol: str,
        side: str,
        qty: float,
        status: str,
        filled_qty: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        payload_json = json.dumps(payload) if payload is not None else None

        self._conn.execute(
            """
            INSERT INTO orders (client_order_id, alpaca_order_id, symbol, side, qty, status, filled_qty, created_ts, updated_ts, last_payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_order_id) DO UPDATE SET
              alpaca_order_id=excluded.alpaca_order_id,
              status=excluded.status,
              filled_qty=excluded.filled_qty,
              updated_ts=excluded.updated_ts,
              last_payload_json=COALESCE(excluded.last_payload_json, orders.last_payload_json);
            """,
            (
                client_order_id,
                alpaca_order_id,
                symbol,
                side,
                float(qty),
                status,
                float(filled_qty),
                now,
                now,
                payload_json,
            ),
        )
        self._conn.commit()
