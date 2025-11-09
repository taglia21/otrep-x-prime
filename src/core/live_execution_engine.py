"""
live_execution_engine.py
Phase VI – Real-Time Execution Engine with broker connectivity and latency-aware routing
"""

import os, time, logging, numpy as np, requests
from datetime import datetime
from core.strategy_registry import get_strategies
from core.ml_optimizer import MLOptimizer
from core.risk_manager import RiskManager
from core.portfolio_manager import PortfolioManager

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# -------------------------------------------------------------------------
# Broker API wrapper (Alpaca REST)
# -------------------------------------------------------------------------
class AlpacaBroker:
    def __init__(self, paper=True):
        base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.key = os.getenv("ALPACA_API_KEY_ID")
        self.secret = os.getenv("ALPACA_API_SECRET_KEY")
        self.url = base
        self.headers = {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json"
        }

    def get_quote(self, symbol):
        r = requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest", headers=self.headers)
        if r.status_code != 200:
            raise RuntimeError(f"Quote fetch failed: {r.text}")
        q = r.json().get("quote", {})
        return float(q.get("ap", q.get("bp", 0)))

    def submit_order(self, symbol, qty, side, order_type="market", tif="day"):
        payload = {
            "symbol": symbol, "qty": qty, "side": side,
            "type": order_type, "time_in_force": tif
        }
        t0 = time.perf_counter()
        r = requests.post(f"{self.url}/v2/orders", json=payload, headers=self.headers)
        latency = time.perf_counter() - t0
        if r.status_code != 200:
            log.error(f"[BROKER] Order failed: {r.text}")
            return None
        data = r.json()
        log.info(f"[BROKER] {side.upper()} {symbol} x{qty} @ {data.get('filled_avg_price','pending')} "
                 f"(latency={latency:.3f}s)")
        return data

# -------------------------------------------------------------------------
# Live execution loop
# -------------------------------------------------------------------------
class LiveExecutionEngine:
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

    def run(self, interval=60):
        log.info("[ENGINE] Starting real-time trading loop …")
        while True:
            try:
                for symbol in self.symbols:
                    price = self.broker.get_quote(symbol)
                    recent_returns = [np.random.normal(0, 0.01) for _ in range(3)]
                    size = self.risk.size_position(symbol, recent_returns)
                    signals = {s.__class__.__name__: s.generate_signal(price)
                               for s in self.strategies}
                    action = self.optimizer.choose_action(signals)

                    if action > 0:
                        self.broker.submit_order(symbol, size, "buy")
                        self.portfolio.update_position(symbol, size, price, "LIVE")
                    elif action < 0 or self.risk.check_exit(symbol, price):
                        self.broker.submit_order(symbol, size, "sell")
                        self.portfolio.update_position(symbol, -size, price, "LIVE")

                self.optimizer.adapt_weights()
                time.sleep(interval)
            except KeyboardInterrupt:
                log.warning("[ENGINE] Stopped manually.")
                break
            except Exception as e:
                log.error(f"[ENGINE] {e}")
                time.sleep(interval)
