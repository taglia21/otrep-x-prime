"""
async_stream_engine.py
Phase VII – Asynchronous Streaming & Multi-Threaded Execution
"""
import os, json, time, asyncio, threading, logging, websockets
import numpy as np
from datetime import datetime
from core.strategy_registry import get_strategies
from core.ml_optimizer import MLOptimizer
from core.risk_manager import RiskManager
from core.portfolio_manager import PortfolioManager
from core.live_execution_engine import AlpacaBroker
from core.safety import assert_not_killed

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# WebSocket consumer for Alpaca data stream
# ---------------------------------------------------------------------
class AlpacaStream:
    def __init__(self, symbols):
        self.key    = os.getenv("ALPACA_API_KEY_ID")
        self.secret  = os.getenv("ALPACA_API_SECRET_KEY")
        self.url    = "wss://stream.data.alpaca.markets/v2/sip"
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

    async def handle_message(self, msg):
        if not msg or not isinstance(msg, list):
            return
        for entry in msg:
            if "T" in entry:  # Trade update
                symbol = entry["S"]
                price  = entry.get("p") or entry.get("ap") or 0
                if price <= 0:
                    continue
                threading.Thread(target=self.process_signal, args=(symbol, price), daemon=True).start()
            elif "success" in entry and entry["msg"] == "authenticated":
                log.info("[STREAM] Authenticated successfully.")
            elif "time" in entry:
                self.last_ping = time.time()

    def process_signal(self, symbol, price):
        try:
            assert_not_killed()
            recent_returns = [np.random.normal(0, 0.01) for _ in range(3)]
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
