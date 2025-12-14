"""
Phase VII – Launch Asynchronous Streaming Engine

Run from repo root:
    python src/core/app_stage07_async.py

Streaming feed selection:
    - Default: IEX (works with standard Alpaca keys)
    - Optional: set ALPACA_DATA_FEED=sip if you have SIP entitlement
    - Override: set ALPACA_STREAM_URL=<wss://...> to force a specific endpoint

Safety:
    Set NO_ORDERS=1 to prevent any order placement.
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
