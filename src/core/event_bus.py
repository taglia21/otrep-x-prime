"""
OTREP-X PRIME — Phase IX: Event Bus / Pub-Sub System
Provides centralized, low-latency messaging between components.
"""

import os
import json
import time
import redis
import threading
import logging
from datetime import datetime, timezone

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CHANNEL = os.getenv("EVENT_CHANNEL", "otrep_events")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [EVENTBUS] %(message)s"
)
logger = logging.getLogger("event_bus")


# ---------------------------------------------------------------------
# Redis Connection
# ---------------------------------------------------------------------
def get_redis():
    while True:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            return r
        except Exception as e:
            logger.error(f"Redis unavailable → {e}")
            time.sleep(2)


# ---------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------
def publish_event(r: redis.Redis, event_type: str, payload: dict):
    """
    Publishes a structured event message to the Redis channel.
    """
    message = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "payload": payload
    }
    r.publish(CHANNEL, json.dumps(message))
    logger.info(f"Published event [{event_type}] → {payload}")


# ---------------------------------------------------------------------
# Subscriber / Listener
# ---------------------------------------------------------------------
def subscribe(r: redis.Redis, callback):
    """
    Listens for incoming messages and routes them to callback.
    """
    pubsub = r.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(CHANNEL)
    logger.info(f"Subscribed to Redis channel '{CHANNEL}'")

    for msg in pubsub.listen():
        try:
            data = json.loads(msg["data"])
            callback(data)
        except Exception as e:
            logger.error(f"Failed to process message → {e}")


# ---------------------------------------------------------------------
# Example Callback
# ---------------------------------------------------------------------
def handle_event(event):
    etype = event.get("type", "Unknown")
    payload = event.get("payload", {})
    logger.info(f"Event Received [{etype}] → {payload}")


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
def main():
    r = get_redis()

    # Start a listener thread so publishing can run concurrently
    listener = threading.Thread(target=subscribe, args=(r, handle_event), daemon=True)
    listener.start()

    # Emit test events every 5 seconds
    while True:
        test_payload = {"price": round(100 + 10 * (time.time() % 1), 2)}
        publish_event(r, "MARKET_TICK", test_payload)
        time.sleep(5)


if __name__ == "__main__":
    main()
