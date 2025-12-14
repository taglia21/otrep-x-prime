"""
Phase VI – Launch Live Execution Engine

Run from repo root:
    python src/core/app_stage06_live.py

Safety:
    Set NO_ORDERS=1 to prevent any order placement (quotes may still be fetched).
"""
import yaml, logging
from core.live_execution_engine import LiveExecutionEngine

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def main():
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    engine = LiveExecutionEngine(cfg)
    engine.run(interval=60)   # poll every 60 s

if __name__ == "__main__":
    main()
