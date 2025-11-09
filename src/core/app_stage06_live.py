"""
Phase VI – Launch Live Execution Engine
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
