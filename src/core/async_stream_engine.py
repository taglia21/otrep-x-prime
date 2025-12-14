"""
async_stream_engine.py
Phase VII – Asynchronous Streaming & Multi-Threaded Execution
"""

import asyncio
import json
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import websockets

from core.alpaca_rest import AlpacaRestClient
from core.strategy_registry import get_strategies
from core.ml_optimizer import MLOptimizer
from core.risk_manager import RiskManager
from core.portfolio_manager import PortfolioManager
from core.safety import assert_not_killed

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


class AlpacaBroker:
    """Minimal broker adapter for legacy async engine.

    Uses the hardened Alpaca REST client. Order submission remains blocked by
    default via `core.safety.guard_order_submission` inside `AlpacaRestClient`.
    """

    def __init__(self, *, paper: bool = True):
        self._client = AlpacaRestClient(paper=paper)

    def submit_order(self, symbol: str, qty: float, side: str) -> None:
        self._client.submit_order(symbol=symbol, qty=qty, side=side, order_type="market", tif="day")

# ---------------------------------------------------------------------
# WebSocket consumer for Alpaca data stream
# ---------------------------------------------------------------------
class AlpacaStream:
    def __init__(self, symbols):
        self.key    = os.getenv("ALPACA_API_KEY_ID")
        self.secret  = os.getenv("ALPACA_API_SECRET_KEY")

        feed = (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower()
        if feed not in {"iex", "sip"}:
            feed = "iex"

        self.url = (os.getenv("ALPACA_STREAM_URL") or f"wss://stream.data.alpaca.markets/v2/{feed}").strip()
        log.info(f"[STREAM] feed={feed} url={self.url}")
        self.symbols = symbols
        self.queue  = asyncio.Queue()
        self.ws     = None
        self.running = True

    async def connect(self):
        async with websockets.connect(self.url) as ws:
            self.ws = ws
            assert_not_killed()
            await ws.send(json.dumps({
                "action": "auth",
                "key": self.key,
                "secret": self.secret
            }))
            await ws.send(json.dumps({
                "action": "subscribe",
                "trades": self.symbols,
                "quotes": self.symbols
            }))
            log.info(f"[STREAM] Subscribed to {self.symbols}")
            async for msg in ws:
                assert_not_killed()
                data = json.loads(msg)
                await self.queue.put(data)
                if not self.running:
                    break

    def stop(self):
        self.running = False

# ---------------------------------------------------------------------
# Core async execution loop with multi-threaded signal processing
# ---------------------------------------------------------------------
class AsyncEngine:
    def __init__(self, config):
        self.cfg = config
        self.strategies = get_strategies(config["strategies"])
        self.optimizer = MLOptimizer(config["strategies"], learning_rate=0.2)
        self.risk = RiskManager(**config["risk_management"])
        self.portfolio = PortfolioManager()
        self.broker = AlpacaBroker(paper=True)
        self.symbols = config["markets"]["US"]["symbols"] + \
                        config["markets"]["EU"]["symbols"] + \
                        config["markets"]["ASIA"]["symbols"]
        self.stream = AlpacaStream(self.symbols)
        self.last_ping = time.time()

        self._price_last: dict[str, float] = {}
        self._returns: dict[str, deque[float]] = {s: deque(maxlen=50) for s in self.symbols}

        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="async-signal")
        self._pending: deque[Future[None]] = deque()
        self._pending_lock = threading.Lock()
        self._max_pending = 64

    async def handle_message(self, msg):
        if not msg or not isinstance(msg, list):
            return
        for entry in msg:
            if "T" in entry:  # Trade update
                symbol = entry["S"]
                price  = entry.get("p") or entry.get("ap") or 0
                if price <= 0:
                    continue

                p = float(price)
                prev = self._price_last.get(symbol)
                if prev is not None and prev > 0:
                    r = p / prev - 1.0
                    self._returns.setdefault(symbol, deque(maxlen=50)).append(float(r))
                self._price_last[symbol] = p

                self._submit_signal(symbol=symbol, price=p)
            elif "success" in entry and entry["msg"] == "authenticated":
                log.info("[STREAM] Authenticated successfully.")
            elif "time" in entry:
                self.last_ping = time.time()

    def _submit_signal(self, *, symbol: str, price: float) -> None:
        with self._pending_lock:
            while self._pending and self._pending[0].done():
                self._pending.popleft()
            if len(self._pending) >= self._max_pending:
                log.warning(f"[ASYNC] backlog_full symbol={symbol} pending={len(self._pending)}")
                return
            fut = self._executor.submit(self.process_signal, symbol, price)
            self._pending.append(fut)

    def process_signal(self, symbol, price):
        try:
            assert_not_killed()
            recent_returns = list(self._returns.get(symbol, deque()))
            size = self.risk.size_position(symbol, recent_returns)
            signals = {s.__class__.__name__: s.generate_signal(price) for s in self.strategies}
            action = self.optimizer.choose_action(signals)
            if action > 0:
                self.broker.submit_order(symbol, size, "buy")
                self.portfolio.update_position(symbol, size, price, "LIVE")
            elif action < 0 or self.risk.check_exit(symbol, price):
                self.broker.submit_order(symbol, size, "sell")
                self.portfolio.update_position(symbol, -size, price, "LIVE")
            self.optimizer.adapt_weights()
        except Exception as e:
            log.error(f"[ASYNC] Signal error: {e}")

    async def run(self):
        log.info("[ENGINE] Starting asynchronous streaming engine …")
        consumer = asyncio.create_task(self.stream.connect())
        while self.stream.running:
            assert_not_killed()
            msg = await self.stream.queue.get()
            await self.handle_message(msg)
            if time.time() - self.last_ping > 30:
                log.warning("[STREAM] No heartbeat, attempting reconnect …")
                self.stream.stop()
                await asyncio.sleep(5)
                self.stream = AlpacaStream(self.symbols)
                consumer = asyncio.create_task(self.stream.connect())
        await consumer
