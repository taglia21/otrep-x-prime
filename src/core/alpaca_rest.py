from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from core.retry import RetryPolicy, parse_retry_after_seconds, retry_call
from core.safety import guard_order_submission
from core.trading_halt import guard_not_halted


@dataclass(frozen=True)
class AlpacaAccount:
    equity: float
    cash: float
    currency: str | None = None


@dataclass(frozen=True)
class AlpacaPosition:
    symbol: str
    qty: float
    avg_entry_price: float | None
    market_value: float | None


@dataclass(frozen=True)
class AlpacaOrder:
    id: str
    client_order_id: str | None
    symbol: str
    side: str
    qty: float
    filled_qty: float
    status: str


class AlpacaRestClient:
    def __init__(self, *, paper: bool = True, timeout_s: float = 10.0):
        self.paper = bool(paper)
        self.timeout_s = float(timeout_s)

        self.trading_base_url = (
            "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"
        )
        self.data_base_url = "https://data.alpaca.markets"

        # Do not print these.
        import os

        self._key = os.getenv("ALPACA_API_KEY_ID")
        self._secret = os.getenv("ALPACA_API_SECRET_KEY")

        self._headers = {
            "APCA-API-KEY-ID": self._key or "",
            "APCA-API-SECRET-KEY": self._secret or "",
            "Content-Type": "application/json",
        }

    def _get(self, url: str, *, params: dict[str, Any] | None = None) -> requests.Response:
        return self._request("GET", url, params=params, payload=None)

    def _post(self, url: str, *, payload: dict[str, Any]) -> requests.Response:
        return self._request("POST", url, params=None, payload=payload)

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None,
        payload: dict[str, Any] | None,
    ) -> requests.Response:
        policy = RetryPolicy()

        def _do() -> requests.Response:
            try:
                return requests.request(
                    method,
                    url,
                    headers=self._headers,
                    params=params,
                    json=payload,
                    timeout=self.timeout_s,
                )
            except requests.exceptions.Timeout as e:
                # Represent a timeout as a synthetic response-like object via raising.
                raise e
            except requests.exceptions.ConnectionError as e:
                raise e

        def _call_with_exception_capture() -> requests.Response:
            try:
                return _do()
            except requests.exceptions.Timeout:
                r = requests.Response()
                r.status_code = 408
                r._content = b"{}"  # type: ignore[attr-defined]
                r.url = url
                return r
            except requests.exceptions.ConnectionError:
                r = requests.Response()
                r.status_code = 503
                r._content = b"{}"  # type: ignore[attr-defined]
                r.url = url
                return r

        def _is_retryable(r: requests.Response) -> bool:
            return int(r.status_code) in policy.retry_http_statuses

        def _retry_after(r: requests.Response) -> float | None:
            if int(r.status_code) != 429:
                return None
            return parse_retry_after_seconds(r.headers.get("Retry-After"))

        return retry_call(
            _call_with_exception_capture,
            policy=policy,
            is_retryable=_is_retryable,
            retry_after_s=_retry_after,
        )

    # ------------------------------ Account state ------------------------------

    def get_account(self) -> AlpacaAccount:
        r = self._get(f"{self.trading_base_url}/v2/account")
        if r.status_code != 200:
            raise RuntimeError(f"account_fetch_failed status={r.status_code}")
        data = r.json()
        return AlpacaAccount(
            equity=float(data.get("equity", 0.0)),
            cash=float(data.get("cash", 0.0)),
            currency=data.get("currency"),
        )

    def get_positions(self) -> list[AlpacaPosition]:
        r = self._get(f"{self.trading_base_url}/v2/positions")
        if r.status_code != 200:
            raise RuntimeError(f"positions_fetch_failed status={r.status_code}")
        items = r.json() if isinstance(r.json(), list) else []
        out: list[AlpacaPosition] = []
        for p in items:
            out.append(
                AlpacaPosition(
                    symbol=str(p.get("symbol")),
                    qty=float(p.get("qty", 0.0)),
                    avg_entry_price=float(p["avg_entry_price"]) if p.get("avg_entry_price") is not None else None,
                    market_value=float(p["market_value"]) if p.get("market_value") is not None else None,
                )
            )
        return out

    def get_open_orders(self) -> list[AlpacaOrder]:
        r = self._get(f"{self.trading_base_url}/v2/orders", params={"status": "open"})
        if r.status_code != 200:
            raise RuntimeError(f"open_orders_fetch_failed status={r.status_code}")
        items = r.json() if isinstance(r.json(), list) else []
        return [self._parse_order(o) for o in items]

    def get_order(self, order_id: str) -> AlpacaOrder:
        r = self._get(f"{self.trading_base_url}/v2/orders/{order_id}")
        if r.status_code != 200:
            raise RuntimeError(f"order_fetch_failed status={r.status_code}")
        return self._parse_order(r.json())

    def _parse_order(self, o: dict[str, Any]) -> AlpacaOrder:
        return AlpacaOrder(
            id=str(o.get("id")),
            client_order_id=o.get("client_order_id"),
            symbol=str(o.get("symbol")),
            side=str(o.get("side")),
            qty=float(o.get("qty", 0.0)),
            filled_qty=float(o.get("filled_qty", 0.0)),
            status=str(o.get("status")),
        )

    # ------------------------------ Market data ------------------------------

    def get_recent_bars(
        self,
        *,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        # Alpaca Stocks Bars v2:
        # GET /v2/stocks/bars?symbols=AAPL&timeframe=1Min&limit=...
        r = self._get(
            f"{self.data_base_url}/v2/stocks/bars",
            params={"symbols": symbol, "timeframe": timeframe, "limit": int(limit)},
        )
        if r.status_code != 200:
            raise RuntimeError(f"bars_fetch_failed status={r.status_code}")
        payload = r.json()
        bars = payload.get("bars", {})
        series = bars.get(symbol) or []
        if not isinstance(series, list):
            return []
        return series

    # ------------------------------ Order submission ------------------------------

    def submit_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        tif: str = "day",
        client_order_id: str | None = None,
    ) -> tuple[AlpacaOrder | None, float, int]:
        # Safety: if the process is halted, never submit.
        guard_not_halted()

        # Safety: hard block unless explicitly enabled.
        guard_order_submission(require_ack=not self.paper)

        payload: dict[str, Any] = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": tif,
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id

        t0 = time.perf_counter()
        r = self._post(f"{self.trading_base_url}/v2/orders", payload=payload)
        latency = time.perf_counter() - t0

        if r.status_code != 200:
            return None, float(latency), int(r.status_code)

        order = self._parse_order(r.json())
        return order, float(latency), int(r.status_code)
