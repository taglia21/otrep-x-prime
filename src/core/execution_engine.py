# src/core/execution_engine.py
import json
import logging
import time
import yaml
import redis
from pathlib import Path

CONFIG_PATH = Path("/app/config/engine.yaml")

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config().get("engine", {})
    redis_host = cfg.get("redis_host", "redis")
    signal_channel = cfg.get("signal_channel", "trade_signals")
    log_channel = cfg.get("log_channel", "execution_log")

    logging.basicConfig(
        level=logging.INFO,
        format="[ENGINE] %(asctime)s %(levelname)s: %(message)s",
    )
    logging.info("Booting execution engine...")
    logging.info(f"Config: redis_host={redis_host} signal_channel={signal_channel} log_channel={log_channel}")

    r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
    pubsub = r.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(signal_channel)
    logging.info("Execution Engine online and awaiting signals...")

    # Simple loop: consume signals and acknowledge to a log list
    while True:
        msg = pubsub.get_message(timeout=1.0)
        if msg and msg.get("type") == "message":
            raw = msg.get("data")
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"raw": raw}
            logging.info(f"Signal received -> {payload}")
            r.rpush(log_channel, json.dumps({"ts": time.time(), "received": payload}))
        time.sleep(0.1)

if __name__ == "__main__":
    main()
