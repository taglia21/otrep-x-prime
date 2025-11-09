"""
Phase VII – Launch Asynchronous Streaming Engine
"""
import yaml, asyncio, logging
from core.async_stream_engine import AsyncEngine

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def main():
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    engine = AsyncEngine(cfg)
    asyncio.run(engine.run())

if __name__ == "__main__":
    main()
