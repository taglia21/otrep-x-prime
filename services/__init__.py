"""
Services Module
===============
Microservice stubs for OTREP-X PRIME.

Components:
- MarketDataService: Cached market data with failover
- DataStaleError: Exception for stale data
- DataFetchError: Exception for fetch failures
"""

from .market_data_service import (
    MarketDataService,
    MarketDataConfig,
    CacheEntry,
    DataStaleError,
    DataFetchError,
    create_market_data_service,
)

__all__ = [
    'MarketDataService',
    'MarketDataConfig',
    'CacheEntry',
    'DataStaleError',
    'DataFetchError',
    'create_market_data_service',
]
