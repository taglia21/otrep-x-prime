"""
Market Data Service - Decoupled Data Layer
===========================================
SPOF mitigation: Caching and failover wrapper for market data.

Features:
- In-memory caching with TTL
- Data staleness detection with DataStaleError
- Automatic failover between Polygon and Alpaca
- Exponential backoff on failures
- Batch fetch capability
- Async-ready design (sync implementation for v1)

Author: OTREP-X Development Team
Date: December 2025
"""

import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import pandas as pd


class DataStaleError(Exception):
    """Raised when data is stale beyond threshold."""
    
    def __init__(self, symbol: str, staleness_seconds: float, threshold_seconds: float):
        self.symbol = symbol
        self.staleness_seconds = staleness_seconds
        self.threshold_seconds = threshold_seconds
        super().__init__(
            f"Data for {symbol} is stale: {staleness_seconds:.1f}s > {threshold_seconds:.1f}s threshold"
        )


class DataFetchError(Exception):
    """Raised when all data sources fail after retries."""
    
    def __init__(self, symbol: str, attempts: int, last_error: Optional[str] = None):
        self.symbol = symbol
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Failed to fetch data for {symbol} after {attempts} attempts. Last error: {last_error}"
        )


@dataclass
class CacheEntry:
    """Cached data entry with timestamp."""
    data: Any
    timestamp: float  # Unix timestamp
    symbol: str
    timeframe: str
    
    def is_stale(self, max_age_seconds: float) -> bool:
        """Check if cache entry is stale."""
        return (time.time() - self.timestamp) > max_age_seconds


@dataclass
class MarketDataConfig:
    """Configuration for market data service."""
    # Cache TTL in seconds (5 minutes for 5-min bars)
    cache_ttl_seconds: float = 300.0
    
    # Staleness threshold (10 minutes per CTO directive)
    staleness_threshold_seconds: float = 600.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Enable caching
    caching_enabled: bool = True


class MarketDataService:
    """
    Decoupled Market Data Service.
    
    Wraps PolygonClient (and AlpacaClient as fallback) to provide:
    - In-memory caching
    - Staleness detection
    - Automatic failover
    - Metrics collection
    
    This removes the Single Point of Failure (SPOF) from MVTTraderLive
    by introducing a resilient data layer.
    
    Usage:
        from api.polygon_client import PolygonClient
        from services.market_data_service import MarketDataService
        
        polygon = PolygonClient()
        service = MarketDataService(primary_client=polygon)
        
        # Get bars (cached if available)
        df = service.get_bars('AAPL', '5Min', limit=50)
        
        # Check health
        if service.is_data_stale('AAPL'):
            # Handle stale data condition
            pass
    """
    
    def __init__(
        self,
        primary_client: Any,
        fallback_client: Optional[Any] = None,
        config: Optional[MarketDataConfig] = None,
        logger: Optional[Callable[[str], None]] = None,
        kill_switch_callback: Optional[Callable[[], None]] = None
    ):
        """
        Initialize market data service.
        
        Args:
            primary_client: Primary data source (e.g., PolygonClient)
            fallback_client: Fallback data source (e.g., AlpacaClient)
            config: Service configuration
            logger: Logging function
            kill_switch_callback: Callback to trigger kill switch on data failure
        """
        self.primary = primary_client
        self.fallback = fallback_client
        self.config = config or MarketDataConfig()
        self.log = logger or print
        self.kill_switch_callback = kill_switch_callback
        
        # Cache: Dict[cache_key, CacheEntry]
        self._cache: Dict[str, CacheEntry] = {}
        
        # Metrics
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._primary_calls: int = 0
        self._fallback_calls: int = 0
        self._failures: int = 0
        self._last_successful_fetch: Dict[str, float] = {}  # symbol -> timestamp
    
    def _cache_key(self, symbol: str, timeframe: str, limit: int) -> str:
        """Generate cache key for a request."""
        return f"{symbol}:{timeframe}:{limit}"
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str = "5Min",
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV bars for a symbol with caching and failover.
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Bar timeframe (e.g., '5Min', '1Hour')
            limit: Maximum number of bars to return
            
        Returns:
            DataFrame with OHLCV data, or None if all sources fail
        """
        cache_key = self._cache_key(symbol, timeframe, limit)
        
        # Check cache first
        if self.config.caching_enabled and cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_stale(self.config.cache_ttl_seconds):
                self._cache_hits += 1
                return entry.data
        
        self._cache_misses += 1
        
        # Try primary source
        df = self._fetch_with_retry(
            self.primary, symbol, timeframe, limit, source_name="primary"
        )
        
        if df is None and self.fallback:
            # Fallback to secondary source
            self.log(f"⚠️ Primary data source failed for {symbol}, trying fallback")
            df = self._fetch_with_retry(
                self.fallback, symbol, timeframe, limit, source_name="fallback"
            )
            if df is not None:
                self._fallback_calls += 1
        
        if df is not None:
            # Update cache
            if self.config.caching_enabled:
                self._cache[cache_key] = CacheEntry(
                    data=df,
                    timestamp=time.time(),
                    symbol=symbol,
                    timeframe=timeframe
                )
            
            # Update last successful fetch timestamp
            self._last_successful_fetch[symbol] = time.time()
            return df
        
        # All sources failed
        self._failures += 1
        self.log(f"❌ All data sources failed for {symbol}")
        
        # Check if we should trigger kill switch
        if self._should_trigger_kill_switch(symbol):
            if self.kill_switch_callback:
                self.kill_switch_callback()
        
        # Return stale cached data if available (better than nothing)
        if cache_key in self._cache:
            self.log(f"⚠️ Returning stale cached data for {symbol}")
            return self._cache[cache_key].data
        
        return None
    
    def _fetch_with_retry(
        self,
        client: Any,
        symbol: str,
        timeframe: str,
        limit: int,
        source_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data with exponential backoff retries.
        
        Backoff formula: delay = base_delay * (2 ^ attempt)
        With jitter to prevent thundering herd.
        """
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                if source_name == "primary":
                    self._primary_calls += 1
                
                df = client.get_bars(symbol=symbol, timeframe=timeframe, limit=limit)
                
                if df is not None and len(df) > 0:
                    return df
                    
            except Exception as e:
                last_error = str(e)
                self.log(f"⚠️ {source_name} fetch attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff with jitter
                    # delay = base * 2^attempt * (0.5 + random(0.5))
                    import random
                    base_delay = self.config.retry_delay_seconds
                    backoff_delay = base_delay * (2 ** attempt) * (0.5 + random.random() * 0.5)
                    # Cap at 30 seconds
                    backoff_delay = min(backoff_delay, 30.0)
                    self.log(f"   Retrying in {backoff_delay:.1f}s...")
                    time.sleep(backoff_delay)
        
        return None
    
    def get_bars_batch(
        self,
        symbols: List[str],
        timeframe: str = "5Min",
        limit: int = 100,
        raise_on_stale: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch bars for multiple symbols with rate limiting.
        
        Args:
            symbols: List of stock ticker symbols
            timeframe: Bar timeframe
            limit: Maximum bars per symbol
            raise_on_stale: If True, raise DataStaleError for stale symbols
            
        Returns:
            Dict mapping symbol -> DataFrame (symbols that failed are excluded)
            
        Raises:
            DataStaleError: If raise_on_stale=True and a symbol is stale
        """
        results: Dict[str, pd.DataFrame] = {}
        failed_symbols: List[str] = []
        
        for symbol in symbols:
            # Rate limit: small delay between requests to avoid 429
            if results:
                time.sleep(0.05)  # 50ms between requests
            
            try:
                df = self.get_bars(symbol, timeframe, limit)
                
                if df is not None and len(df) > 0:
                    results[symbol] = df
                    
                    # Check staleness if requested
                    if raise_on_stale and self.is_data_stale(symbol):
                        staleness = self.get_staleness_seconds(symbol)
                        raise DataStaleError(
                            symbol=symbol,
                            staleness_seconds=staleness,
                            threshold_seconds=self.config.staleness_threshold_seconds
                        )
                else:
                    failed_symbols.append(symbol)
                    
            except DataStaleError:
                raise  # Re-raise staleness errors
            except Exception as e:
                self.log(f"❌ Failed to fetch {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            self.log(f"⚠️ Batch fetch: {len(results)}/{len(symbols)} succeeded, "
                    f"failed: {failed_symbols[:5]}{'...' if len(failed_symbols) > 5 else ''}")
        
        return results
    
    def get_bars_strict(
        self,
        symbol: str,
        timeframe: str = "5Min",
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get bars with strict staleness checking.
        
        Raises DataStaleError if data is stale, never returns stale cache.
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Bar timeframe
            limit: Maximum bars
            
        Returns:
            DataFrame with fresh data
            
        Raises:
            DataStaleError: If data is stale beyond threshold
            DataFetchError: If all sources fail and no cache available
        """
        df = self.get_bars(symbol, timeframe, limit)
        
        if df is None:
            raise DataFetchError(
                symbol=symbol,
                attempts=self.config.max_retries,
                last_error="All sources failed"
            )
        
        # Check if this is stale cached data
        staleness = self.get_staleness_seconds(symbol)
        if staleness > self.config.staleness_threshold_seconds:
            raise DataStaleError(
                symbol=symbol,
                staleness_seconds=staleness,
                threshold_seconds=self.config.staleness_threshold_seconds
            )
        
        return df
        
        return None
    
    def is_data_stale(self, symbol: str) -> bool:
        """
        Check if data for a symbol is stale.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            True if data is stale (no update in staleness_threshold_seconds)
        """
        if symbol not in self._last_successful_fetch:
            return True
        
        age = time.time() - self._last_successful_fetch[symbol]
        return age > self.config.staleness_threshold_seconds
    
    def get_staleness_seconds(self, symbol: str) -> float:
        """Get data staleness in seconds for a symbol."""
        if symbol not in self._last_successful_fetch:
            return float('inf')
        return time.time() - self._last_successful_fetch[symbol]
    
    def _should_trigger_kill_switch(self, symbol: str) -> bool:
        """Check if data failure should trigger kill switch."""
        # Trigger if all symbols are stale
        all_stale = all(
            self.is_data_stale(s) 
            for s in self._last_successful_fetch.keys()
        ) if self._last_successful_fetch else True
        
        return all_stale and self._failures >= self.config.max_retries
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            symbol: Specific symbol to clear, or None for all
        """
        if symbol:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{symbol}:")]
            for k in keys_to_remove:
                del self._cache[k]
        else:
            self._cache.clear()
    
    def get_metrics(self) -> Dict:
        """Get service metrics for monitoring."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': hit_rate,
            'primary_calls': self._primary_calls,
            'fallback_calls': self._fallback_calls,
            'failures': self._failures,
            'cached_symbols': len(self._cache),
            'staleness_by_symbol': {
                s: self.get_staleness_seconds(s)
                for s in self._last_successful_fetch
            }
        }
    
    def health_check(self) -> Dict:
        """Perform health check on data sources."""
        result = {
            'primary_available': False,
            'fallback_available': False,
            'overall_healthy': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test primary
        try:
            test_df = self.primary.get_bars(symbol='SPY', timeframe='5Min', limit=1)
            result['primary_available'] = test_df is not None and len(test_df) > 0
        except Exception:
            pass
        
        # Test fallback
        if self.fallback:
            try:
                test_df = self.fallback.get_bars(symbol='SPY', timeframe='5Min', limit=1)
                result['fallback_available'] = test_df is not None and len(test_df) > 0
            except Exception:
                pass
        
        result['overall_healthy'] = result['primary_available'] or result['fallback_available']
        return result


def create_market_data_service(
    polygon_client: Any,
    alpaca_client: Optional[Any] = None,
    config_dict: Optional[Dict] = None,
    logger: Optional[Callable] = None
) -> MarketDataService:
    """
    Factory function to create MarketDataService from config.
    
    Args:
        polygon_client: PolygonClient instance
        alpaca_client: Optional AlpacaClient for fallback
        config_dict: Optional config dictionary
        logger: Optional logger function
        
    Returns:
        Configured MarketDataService instance
    """
    config = MarketDataConfig()
    
    if config_dict:
        system_cfg = config_dict.get('SYSTEM', {})
        config.cache_ttl_seconds = system_cfg.get('DATA_CACHE_TTL', 300.0)
        config.staleness_threshold_seconds = system_cfg.get('DATA_STALE_THRESHOLD', 600.0)
    
    return MarketDataService(
        primary_client=polygon_client,
        fallback_client=alpaca_client,
        config=config,
        logger=logger
    )
