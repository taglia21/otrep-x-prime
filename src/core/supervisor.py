"""
OTREP-X PRIME — Phase IX-B
Supervisor ↔ EventBus Integration
Publishes heartbeat & system status events via Redis Pub/Sub.
"""

import os
import time
import json
import redis
import logging
from datetime import datetime, timezone

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
HEARTBEAT_KEY = "otrep_heartbeat"
EVENT_CHANNEL = os.getenv("EVENT_CHANNEL", "otrep_events")
INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", 5))

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [SUPERVISOR] %(message)s"
)
logger = logging.getLogger("supervisor")


# ---------------------------------------------------------------------
# Redis Connection Helper
# ---------------------------------------------------------------------
def get_redis():
    while True:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            return r
        except Exception as e:
            logger.error(f"Redis connection failed → {e}")
            time.sleep(2)


# ---------------------------------------------------------------------
# Heartbeat Publisher
# ---------------------------------------------------------------------
def publish_heartbeat(r: redis.Redis):
    heartbeat = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "healthy",
        "active_strategies": ["Stoichiometric", "Momentum", "MeanReversion"]
    }
    # Update KV store
    r.set(HEARTBEAT_KEY, json.dumps(heartbeat))

    # Publish heartbeat event
    event = {
        "timestamp": heartbeat["timestamp"],
        "type": "SYSTEM_HEARTBEAT",
        "payload": heartbeat
    }
    r.publish(EVENT_CHANNEL, json.dumps(event))

    logger.info(f"Heartbeat broadcast → {heartbeat}")


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
def main():
    r = get_redis()
    logger.info("Supervisor event broadcaster started.")

    while True:
        try:
            publish_heartbeat(r)
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            logger.warning("Supervisor manually stopped.")
            break
        except Exception as e:
            logger.error(f"Error in supervisor loop → {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
