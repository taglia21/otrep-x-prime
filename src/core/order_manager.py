from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.alpaca_rest import AlpacaOrder, AlpacaRestClient
from core.order_journal import JournalOrder, OrderJournal


@dataclass(frozen=True)
class OrderDecision:
    submitted: bool
    reason: str
    client_order_id: str
    alpaca_order_id: str | None


_FINAL_STATUSES = {"filled", "canceled", "rejected", "expired"}


def deterministic_client_order_id(
    *,
    strategy: str,
    symbol: str,
    side: str,
    qty: float,
    order_type: str,
    tif: str,
    bucket_seconds: int = 60,
    now_ts: float | None = None,
) -> str:
    ts = float(time.time() if now_ts is None else now_ts)
    # Treat exact boundaries as belonging to the *previous* bucket so
    # (60.0, 61.0) with bucket_seconds=60 fall into different buckets.
    bucket = int((ts - 1e-9) // bucket_seconds)
    payload = f"{strategy}|{bucket}|{symbol}|{side}|{qty}|{order_type}|{tif}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OrderManager:
    def __init__(
        self,
        *,
        alpaca: AlpacaRestClient,
        journal: OrderJournal,
    ):
        self.alpaca = alpaca
        self.journal = journal

    def submit_order(
        self,
        *,
        strategy: str,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        tif: str = "day",
        bucket_seconds: int = 60,
        now_ts: float | None = None,
    ) -> OrderDecision:
        client_order_id = deterministic_client_order_id(
            strategy=strategy,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            tif=tif,
            bucket_seconds=bucket_seconds,
            now_ts=now_ts,
        )

        existing = self.journal.get(client_order_id)
        if existing is not None:
            return OrderDecision(
                submitted=False,
                reason=f"deduped_existing status={existing.status}",
                client_order_id=client_order_id,
                alpaca_order_id=existing.alpaca_order_id,
            )

        # Broker-side dedupe: if already open on broker, skip.
        try:
            open_orders = self.alpaca.get_open_orders()
            for o in open_orders:
                if o.client_order_id == client_order_id:
                    self.journal.upsert(
                        client_order_id=client_order_id,
                        alpaca_order_id=o.id,
                        symbol=o.symbol,
                        side=o.side,
                        qty=o.qty,
                        status=o.status,
                        filled_qty=o.filled_qty,
                        payload=None,
                    )
                    return OrderDecision(
                        submitted=False,
                        reason="deduped_broker_open",
                        client_order_id=client_order_id,
                        alpaca_order_id=o.id,
                    )
        except Exception:
            # If open orders fetch fails, do not attempt to be clever; proceed to submit and let safety/risk gate.
            pass

        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": tif,
            "client_order_id": client_order_id,
        }

        order, latency, status = self.alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            tif=tif,
            client_order_id=client_order_id,
        )

        if order is None:
            self.journal.upsert(
                client_order_id=client_order_id,
                alpaca_order_id=None,
                symbol=symbol,
                side=side,
                qty=qty,
                status=f"submit_failed_{status}",
                filled_qty=0.0,
                payload=payload,
            )
            return OrderDecision(
                submitted=False,
                reason=f"submit_failed status={status}",
                client_order_id=client_order_id,
                alpaca_order_id=None,
            )

        self.journal.upsert(
            client_order_id=client_order_id,
            alpaca_order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            status=order.status,
            filled_qty=order.filled_qty,
            payload=payload,
        )

        return OrderDecision(
            submitted=True,
            reason=f"submitted latency_s={latency:.3f}",
            client_order_id=client_order_id,
            alpaca_order_id=order.id,
        )

    def poll_and_update(self, *, alpaca_order_id: str) -> AlpacaOrder:
        o = self.alpaca.get_order(alpaca_order_id)
        if o.client_order_id:
            self.journal.upsert(
                client_order_id=o.client_order_id,
                alpaca_order_id=o.id,
                symbol=o.symbol,
                side=o.side,
                qty=o.qty,
                status=o.status,
                filled_qty=o.filled_qty,
                payload=None,
            )
        return o
