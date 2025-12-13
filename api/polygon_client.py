"""
Polygon API Client - DATA RETRIEVAL
====================================
This client handles all market data retrieval via Polygon.io.
Returns pandas DataFrames for vectorized analysis.

Author: OTREP-X Development Team
Lead Engineer: Gemini AI
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import numpy as np

try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False


class PolygonClient:
    """
    Polygon.io API client for market data retrieval.
    
    All methods return pandas DataFrames for efficient vectorized analysis.
    Falls back to Alpaca data API if Polygon is unavailable.
    """
    
    def __init__(self, alpaca_fallback_url: Optional[str] = None):
        """
        Initialize Polygon client with API key from environment.
        
        Args:
            alpaca_fallback_url: Alpaca data URL for fallback if Polygon unavailable
        """
        self.api_key = os.getenv('POLYGON_API_KEY', '')
        self.alpaca_fallback_url = alpaca_fallback_url or 'https://data.alpaca.markets'
        # Support both naming conventions for Alpaca keys
        self.alpaca_key = os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_API_KEY_ID', '')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET_KEY', '')
        
        self.client = None
        if POLYGON_AVAILABLE and self.api_key:
            self.client = RESTClient(self.api_key)
    
    @property
    def is_available(self) -> bool:
        """Check if Polygon API is available and configured."""
        return self.client is not None
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str = '5Min',
        limit: int = 100,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Bar timeframe ('1Min', '5Min', '1Hour', '1Day')
            limit: Number of bars to fetch
            end_date: End date for historical data (default: now)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if self.client:
            return self._get_polygon_bars(symbol, timeframe, limit, end_date)
        else:
            return self._get_alpaca_bars(symbol, timeframe, limit)
    
    def _get_polygon_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Fetch bars from Polygon API."""
        # Parse timeframe to Polygon format
        multiplier, timespan = self._parse_timeframe(timeframe)
        
        end = end_date or datetime.now()
        # Calculate start date based on limit and timeframe
        if timespan == 'minute':
            start = end - timedelta(minutes=limit * multiplier * 2)  # Extra buffer
        elif timespan == 'hour':
            start = end - timedelta(hours=limit * multiplier * 2)
        else:  # day
            start = end - timedelta(days=limit * multiplier * 2)
        
        try:
            aggs = list(self.client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start.strftime('%Y-%m-%d'),
                to=end.strftime('%Y-%m-%d'),
                limit=limit
            ))
            
            if not aggs:
                return pd.DataFrame()
            
            df = pd.DataFrame([{
                'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                'open': float(agg.open),
                'high': float(agg.high),
                'low': float(agg.low),
                'close': float(agg.close),
                'volume': int(agg.volume)
            } for agg in aggs])
            
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df.tail(limit)
            
        except Exception as e:
            print(f"Polygon API error for {symbol}: {e}")
            return self._get_alpaca_bars(symbol, timeframe, limit)
    
    def _get_alpaca_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """Fallback to Alpaca data API."""
        import requests
        
        headers = {
            'APCA-API-KEY-ID': self.alpaca_key,
            'APCA-API-SECRET-KEY': self.alpaca_secret
        }
        
        # Convert timeframe to Alpaca format
        alpaca_timeframe = self._convert_to_alpaca_timeframe(timeframe)
        
        # Calculate start date for historical data
        from datetime import datetime, timedelta
        end_date = datetime.now()
        
        # Estimate days needed based on limit and timeframe
        if 'Day' in alpaca_timeframe or 'day' in alpaca_timeframe:
            start_date = end_date - timedelta(days=limit + 60)  # Extra buffer for weekends/holidays
        elif 'Hour' in alpaca_timeframe or 'hour' in alpaca_timeframe:
            start_date = end_date - timedelta(hours=limit * 2)
        else:  # Minutes
            start_date = end_date - timedelta(days=max(5, limit // 78))  # ~78 5-min bars per day
        
        url = f'{self.alpaca_fallback_url}/v2/stocks/{symbol}/bars'
        params = {
            'timeframe': alpaca_timeframe,
            'start': start_date.strftime('%Y-%m-%dT00:00:00Z'),
            'end': end_date.strftime('%Y-%m-%dT23:59:59Z'),
            'limit': limit,
            'adjustment': 'split',
            'feed': 'iex'  # Use IEX feed for free tier compatibility
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            bars = data.get('bars', [])
            
            if not bars:
                return pd.DataFrame()
            
            df = pd.DataFrame([{
                'timestamp': pd.to_datetime(bar['t']),
                'open': float(bar['o']),
                'high': float(bar['h']),
                'low': float(bar['l']),
                'close': float(bar['c']),
                'volume': int(bar['v'])
            } for bar in bars])
            
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df.tail(limit)
            
        except Exception as e:
            print(f"Alpaca data API error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _convert_to_alpaca_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Alpaca API format."""
        tf = timeframe.lower().replace('min', 'min').replace('hour', 'hour').replace('day', 'day')
        
        # Alpaca accepts: 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day, 1Week, 1Month
        if 'min' in tf:
            num = int(tf.replace('min', ''))
            return f'{num}Min'
        elif 'hour' in tf:
            num = int(tf.replace('hour', '')) if tf != 'hour' else 1
            return f'{num}Hour'
        elif 'day' in tf:
            return '1Day'
        elif 'week' in tf:
            return '1Week'
        else:
            return '1Day'  # Default to daily
    
    def _parse_timeframe(self, timeframe: str) -> tuple:
        """Parse timeframe string to Polygon format."""
        timeframe = timeframe.lower()
        
        if 'min' in timeframe:
            multiplier = int(timeframe.replace('min', ''))
            return multiplier, 'minute'
        elif 'hour' in timeframe:
            multiplier = int(timeframe.replace('hour', ''))
            return multiplier, 'hour'
        elif 'day' in timeframe:
            multiplier = int(timeframe.replace('day', '')) if timeframe != 'day' else 1
            return multiplier, 'day'
        else:
            # Default to 5 minute
            return 5, 'minute'
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = '1Day',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical bar data for backtesting and optimization.
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Bar timeframe ('1Min', '5Min', '1Hour', '1Day')
            start_date: Start date string (YYYY-MM-DD format)
            end_date: End date string (YYYY-MM-DD format)
            
        Returns:
            DataFrame with OHLCV data indexed by datetime
        """
        # Calculate days between dates for limit parameter
        if start_date and end_date:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            days_diff = (end - start).days
            end_datetime = end
        else:
            days_diff = 180  # Default 6 months
            end_datetime = datetime.now()
        
        # Estimate bars needed based on timeframe
        if 'min' in timeframe.lower():
            # Assume 6.5 trading hours per day
            mins = int(timeframe.lower().replace('min', ''))
            limit = int(days_diff * (6.5 * 60) / mins)
        elif 'hour' in timeframe.lower():
            hours = int(timeframe.lower().replace('hour', ''))
            limit = int(days_diff * 6.5 / hours)
        else:  # Daily
            limit = days_diff
        
        # Cap limit to reasonable maximum
        limit = min(limit, 5000)
        
        return self.get_bars(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            end_date=end_datetime
        )
    
    def get_daily_bars(
        self,
        symbol: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for correlation analysis.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of trading days to fetch
            
        Returns:
            DataFrame with daily OHLCV data
        """
        return self.get_bars(symbol, '1Day', days)
    
    def calculate_correlation(
        self,
        symbol: str,
        market_symbol: str = 'SPY',
        lookback_days: int = 30
    ) -> float:
        """
        Calculate correlation between symbol and market proxy.
        
        Uses log returns for proper correlation calculation.
        
        Args:
            symbol: Stock ticker symbol
            market_symbol: Market proxy symbol (default: SPY)
            lookback_days: Days of data for calculation
            
        Returns:
            Pearson correlation coefficient [-1, 1]
        """
        try:
            # Fetch daily data for both
            symbol_df = self.get_daily_bars(symbol, lookback_days + 5)
            market_df = self.get_daily_bars(market_symbol, lookback_days + 5)
            
            if symbol_df.empty or market_df.empty:
                return 0.0
            
            if len(symbol_df) < 10 or len(market_df) < 10:
                return 0.0
            
            # Calculate log returns
            symbol_returns = np.log(symbol_df['close'] / symbol_df['close'].shift(1)).dropna()
            market_returns = np.log(market_df['close'] / market_df['close'].shift(1)).dropna()
            
            # Align by date
            common_dates = symbol_returns.index.intersection(market_returns.index)
            if len(common_dates) < 10:
                return 0.0
            
            symbol_aligned = symbol_returns.loc[common_dates]
            market_aligned = market_returns.loc[common_dates]
            
            # Calculate Pearson correlation
            correlation = symbol_aligned.corr(market_aligned)
            
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            print(f"Error calculating correlation for {symbol}: {e}")
            return 0.0
    
    def filter_by_correlation(
        self,
        symbols: List[str],
        market_symbol: str = 'SPY',
        min_correlation: float = 0.75,
        lookback_days: int = 30
    ) -> List[str]:
        """
        Filter symbols by minimum correlation to market proxy.
        
        This is a simplified "market topology" filter - only trade
        symbols that move with the broader market.
        
        Args:
            symbols: List of symbols to filter
            market_symbol: Market proxy for correlation
            min_correlation: Minimum required correlation
            lookback_days: Days for correlation calculation
            
        Returns:
            List of symbols meeting correlation threshold
        """
        filtered = []
        
        for symbol in symbols:
            if symbol == market_symbol:
                filtered.append(symbol)
                continue
                
            corr = self.calculate_correlation(
                symbol, market_symbol, lookback_days
            )
            
            if abs(corr) >= min_correlation:
                filtered.append(symbol)
                print(f"✓ {symbol}: correlation = {corr:.3f}")
            else:
                print(f"✗ {symbol}: correlation = {corr:.3f} (below {min_correlation})")
        
        return filtered
