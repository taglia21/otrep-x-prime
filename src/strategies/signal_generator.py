# src/strategies/signal_generator.py
import json
import random
import time
import redis
import logging

r = redis.Redis(host="redis", port=6379, decode_responses=True)
signal_channel = "trade_signals"

logging.basicConfig(
    level=logging.INFO,
    format="[SIGNAL-GEN] %(asctime)s %(levelname)s: %(message)s",
)

SYMBOLS = ["AAPL", "SPY", "MSFT", "NVDA", "TSLA", "GOOG"]

def generate_signal():
    sym = random.choice(SYMBOLS)
    side = random.choice(["BUY", "SELL"])
    qty = random.choice([1, 5, 10, 25, 50])
    price = round(random.uniform(100, 600), 2)
    return {
        "symbol": sym,
        "side": side,
        "qty": qty,
        "price": price,
        "timestamp": time.time(),
    }

def main():
    logging.info("Signal generator online, publishing to Redis channel: trade_signals")
    while True:
        signal = generate_signal()
        r.publish(signal_channel, json.dumps(signal))
        logging.info(f"Published: {signal}")
        time.sleep(random.uniform(2, 5))

if __name__ == "__main__":
    main()
