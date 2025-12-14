from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Protocol, Sequence

from core.order_intent_journal import OrderIntent, OrderIntentJournal, OrderIntentStore
from core.order_journal import OrderJournal
from core.trading_halt import guard_not_halted


@dataclass(frozen=True)
class OrderDecision:
    submitted: bool
    reason: str
    client_order_id: str
    alpaca_order_id: str | None


_FINAL_STATUSES = {"filled", "canceled", "rejected", "expired"}


class OrderLike(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def client_order_id(self) -> str | None: ...

    @property
    def symbol(self) -> str: ...

    @property
    def side(self) -> str: ...

    @property
    def qty(self) -> float: ...

    @property
    def filled_qty(self) -> float: ...

    @property
    def status(self) -> str: ...


class AlpacaLike(Protocol):
    def get_open_orders(self) -> Sequence[OrderLike]:
        ...

    def submit_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        tif: str = "day",
        client_order_id: str | None = None,
    ) -> tuple[OrderLike | None, float, int]:
        ...

    def get_order(self, order_id: str) -> OrderLike:
        ...


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

    # Deterministic UUID5 based client order id.
    # This avoids accidental collisions while remaining stable across restarts.
    payload = f"otrep-x|{strategy}|{bucket}|{symbol}|{side}|{qty}|{order_type}|{tif}"
    u = uuid.uuid5(uuid.NAMESPACE_URL, payload)
    return u.hex


class OrderManager:
    def __init__(
        self,
        *,
        alpaca: AlpacaLike,
        journal: OrderJournal,
        intent_journal: OrderIntentJournal | None = None,
    ):
        self.alpaca = alpaca
        self.journal = journal
        self.intent_journal = intent_journal
        self.intent_store: OrderIntentStore | None = None

        if self.intent_journal is not None:
            self.intent_store = OrderIntentStore(journal=self.intent_journal)
            self.intent_store.load()
            # Replay: rebuild minimal intent state in SQLite so restart/reconnect
            # won't double-submit when the submission outcome is unknown.
            for e in self.intent_journal.replay():
                if e.get("event") != "order_intent":
                    continue
                coid = e.get("client_order_id")
                if not isinstance(coid, str) or not coid:
                    continue
                if self.journal.get(coid) is not None:
                    continue
                sym = e.get("symbol")
                side = e.get("side")
                qty = e.get("qty")
                if not isinstance(sym, str) or not isinstance(side, str):
                    continue
                if not isinstance(qty, (int, float, str)):
                    continue
                try:
                    self.journal.upsert(
                        client_order_id=coid,
                        alpaca_order_id=None,
                        symbol=sym,
                        side=side,
                        qty=float(qty),
                        status="intent_recorded",
                        filled_qty=0.0,
                        payload=None,
                    )
                except Exception:
                    continue

    def reconcile_open_orders(self) -> None:
        """Read-only broker reconciliation for idempotency.

        Upserts broker open orders into the local SQLite journal, keyed by client_order_id.
        """
        open_orders = self.alpaca.get_open_orders()
        for o in open_orders:
            if not o.client_order_id:
                continue
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
        guard_not_halted()

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

        # Persist the intent *before* any submission attempt.
        intent_ts = float(time.time() if now_ts is None else now_ts)
        if self.intent_journal is not None:
            intent = OrderIntent(
                client_order_id=client_order_id,
                strategy=strategy,
                symbol=symbol,
                side=side,
                qty=float(qty),
                order_type=order_type,
                tif=tif,
                ts=intent_ts,
            )
            self.intent_journal.append_intent(intent=intent)
            if self.intent_store is not None:
                self.intent_store.register(intent=intent)

        # Also store an intent marker in the SQLite journal.
        self.journal.upsert(
            client_order_id=client_order_id,
            alpaca_order_id=None,
            symbol=symbol,
            side=side,
            qty=float(qty),
            status="intent_recorded",
            filled_qty=0.0,
            payload=None,
        )

        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": tif,
            "client_order_id": client_order_id,
        }

        try:
            order, latency, status = self.alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                tif=tif,
                client_order_id=client_order_id,
            )
        except Exception as e:
            self.journal.upsert(
                client_order_id=client_order_id,
                alpaca_order_id=None,
                symbol=symbol,
                side=side,
                qty=float(qty),
                status=f"submit_exception_{type(e).__name__}",
                filled_qty=0.0,
                payload=payload,
            )
            return OrderDecision(
                submitted=False,
                reason=f"submit_exception type={type(e).__name__}",
                client_order_id=client_order_id,
                alpaca_order_id=None,
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

    def poll_and_update(self, *, alpaca_order_id: str) -> OrderLike:
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
