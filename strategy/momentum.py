"""
Hybrid Strategy - Momentum + Mean Reversion
============================================
Combines price momentum with Bollinger Band mean reversion
for a robust multi-alpha signal generation system.

Author: OTREP-X Development Team
Lead Engineer: Gemini AI
Phase III Implementation
"""

import time
from typing import List, Tuple
import pandas as pd
import numpy as np


class HybridStrategy:
    """
    Hybrid trading strategy combining:
    1. Price Momentum with adaptive lookback
    2. Mean Reversion based on Bollinger Bands
    
    Features:
    - Vectorized calculations using pandas/numpy
    - Configurable weights for each alpha signal
    - Adaptive lookback based on market volatility
    """
    
    def __init__(
        self,
        # Momentum parameters
        momentum_lookback: int = 20,
        trend_lookback: int = 15,
        signal_threshold: float = 0.15,
        neutral_threshold: float = 0.05,
        momentum_weight: float = 0.5,
        trend_weight: float = 0.0,
        # Adaptive parameters
        adaptive_enabled: bool = True,
        high_vol_lookback: int = 10,
        low_vol_lookback: int = 30,
        vol_multiplier: float = 1.5,
        # Mean Reversion parameters
        mean_reversion_enabled: bool = True,
        mean_reversion_weight: float = 0.5,
        mean_reversion_lookback: int = 20,
        bb_std_dev_multiplier: float = 2.0,
        reversion_threshold: float = 0.01
    ):
        """
        Initialize hybrid strategy with momentum and mean reversion parameters.
        
        Args:
            momentum_lookback: Default lookback for momentum calculation
            trend_lookback: Lookback for trend calculation
            signal_threshold: Signal strength for trade entry
            neutral_threshold: Signal strength below which to exit
            momentum_weight: Weight for momentum component
            trend_weight: Weight for trend component
            adaptive_enabled: Enable adaptive lookback
            high_vol_lookback: Shorter lookback for high volatility
            low_vol_lookback: Longer lookback for low volatility
            vol_multiplier: Volatility threshold multiplier
            mean_reversion_enabled: Enable mean reversion signal
            mean_reversion_weight: Weight for mean reversion component
            mean_reversion_lookback: Lookback for Bollinger Bands
            bb_std_dev_multiplier: Bollinger Band width factor
            reversion_threshold: Minimum signal for mean reversion
        """
        # Momentum parameters
        self.momentum_lookback = momentum_lookback
        self.trend_lookback = trend_lookback
        self.signal_threshold = signal_threshold
        self.neutral_threshold = neutral_threshold
        self.momentum_weight = momentum_weight
        self.trend_weight = trend_weight
        
        # Adaptive parameters
        self.adaptive_enabled = adaptive_enabled
        self.high_vol_lookback = high_vol_lookback
        self.low_vol_lookback = low_vol_lookback
        self.vol_multiplier = vol_multiplier
        
        # Mean Reversion parameters
        self.mean_reversion_enabled = mean_reversion_enabled
        self.mean_reversion_weight = mean_reversion_weight
        self.mean_reversion_lookback = mean_reversion_lookback
        self.bb_std_dev_multiplier = bb_std_dev_multiplier
        self.reversion_threshold = reversion_threshold
        
        # Volatility tracking
        self.volatility_history: List[float] = []
        
        # Latency profiling (Phase IV)
        self._last_execution_time_ms: float = 0.0
        self._execution_times: List[float] = []
    
    @property
    def last_execution_time_ms(self) -> float:
        """Return the duration of the last calculate_signal call in milliseconds."""
        return self._last_execution_time_ms
    
    @property
    def avg_execution_time_ms(self) -> float:
        """Return the average execution time across all calls in milliseconds."""
        if not self._execution_times:
            return 0.0
        return sum(self._execution_times) / len(self._execution_times)
    
    def reset_profiling(self) -> None:
        """Reset profiling statistics."""
        self._last_execution_time_ms = 0.0
        self._execution_times = []
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        Calculate volatility as standard deviation of log returns.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Volatility (std dev of log returns)
        """
        if len(df) < 2:
            return 0.0
        
        log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        if len(log_returns) == 0:
            return 0.0
        
        return float(log_returns.std())
    
    def get_adaptive_lookback(self, df: pd.DataFrame) -> int:
        """
        Determine lookback period based on current volatility regime.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Adaptive lookback period
        """
        if not self.adaptive_enabled:
            return self.momentum_lookback
        
        if len(df) < 50:
            return self.momentum_lookback
        
        current_vol = self.calculate_volatility(df.tail(20))
        historical_vol = self.calculate_volatility(df.iloc[-50:-20])
        
        self.volatility_history.append(current_vol)
        if len(self.volatility_history) > 100:
            self.volatility_history = self.volatility_history[-100:]
        
        if historical_vol > 0:
            vol_ratio = current_vol / historical_vol
            if vol_ratio >= self.vol_multiplier:
                return self.high_vol_lookback
            elif vol_ratio <= 1.0 / self.vol_multiplier:
                return self.low_vol_lookback
        
        return self.momentum_lookback
    
    def calculate_momentum_signal(self, df: pd.DataFrame) -> Tuple[float, int]:
        """
        Calculate momentum signal from price data.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Tuple of (momentum_signal, lookback_used)
        """
        lookback = self.get_adaptive_lookback(df)
        
        if len(df) < lookback:
            return 0.0, lookback
        
        closes = df['close'].tail(lookback)
        current = closes.iloc[-1]
        average = closes.mean()
        
        if average == 0:
            return 0.0, lookback
        
        momentum = (current - average) / average
        
        # Trend component
        recent_period = min(5, lookback // 4)
        if recent_period > 0 and lookback > recent_period:
            recent = closes.tail(recent_period).mean()
            older = closes.head(lookback - recent_period).mean()
            trend = (recent - older) / older if older != 0 else 0.0
        else:
            trend = 0.0
        
        # Combine momentum and trend
        raw_signal = (
            self.momentum_weight * momentum +
            self.trend_weight * trend
        ) * 10  # Scale to approximately [-1, 1]
        
        return float(np.clip(raw_signal, -1.0, 1.0)), lookback
    
    def calculate_mean_reversion_signal(self, df: pd.DataFrame) -> float:
        """
        Calculate mean reversion signal based on Bollinger Bands.
        
        Formula: Signal_MR = -(Close - SMA) / (STD * BB_STD_DEV_MULTIPLIER)
        
        When price is above SMA -> negative signal (expect reversion down)
        When price is below SMA -> positive signal (expect reversion up)
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Mean reversion signal in range [-1.0, 1.0]
        """
        if not self.mean_reversion_enabled:
            return 0.0
        
        if len(df) < self.mean_reversion_lookback:
            return 0.0
        
        closes = df['close'].tail(self.mean_reversion_lookback + 1)
        
        # Calculate SMA and STD using vectorized rolling operations
        sma = closes.rolling(window=self.mean_reversion_lookback).mean().iloc[-1]
        std = closes.rolling(window=self.mean_reversion_lookback).std().iloc[-1]
        
        if sma == 0 or std == 0 or np.isnan(sma) or np.isnan(std):
            return 0.0
        
        current_price = closes.iloc[-1]
        
        # Mean reversion signal formula (per Gemini spec):
        # Signal_MR = -(Close - SMA) / (STD * BB_STD_DEV_MULTIPLIER)
        # Negative sign: price above SMA -> negative signal (SELL)
        #                price below SMA -> positive signal (BUY)
        signal = -(current_price - sma) / (std * self.bb_std_dev_multiplier)
        
        return float(np.clip(signal, -1.0, 1.0))
    
    def calculate_signal(self, df: pd.DataFrame) -> Tuple[float, int, float]:
        """
        Calculate combined hybrid signal from momentum and mean reversion.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Tuple of (combined_signal, lookback_used, volatility)
        """
        # Start timing for latency profiling
        start_time = time.perf_counter()
        
        # Calculate individual signals
        momentum_signal, lookback_used = self.calculate_momentum_signal(df)
        reversion_signal = self.calculate_mean_reversion_signal(df)
        
        # Combine signals with weights
        # Note: momentum_weight here is just for the momentum component
        # The overall weights are applied to combine momentum and reversion
        total_weight = self.momentum_weight + self.mean_reversion_weight
        
        if total_weight > 0:
            combined_signal = (
                (momentum_signal * self.momentum_weight) +
                (reversion_signal * self.mean_reversion_weight)
            ) / total_weight
        else:
            combined_signal = 0.0
        
        # Clip to [-1, 1]
        combined_signal = float(np.clip(combined_signal, -1.0, 1.0))
        
        # Calculate volatility for logging
        volatility = self.calculate_volatility(df.tail(20))
        
        # Record execution time
        end_time = time.perf_counter()
        self._last_execution_time_ms = (end_time - start_time) * 1000.0
        self._execution_times.append(self._last_execution_time_ms)
        
        # Keep only last 1000 measurements to avoid memory growth
        if len(self._execution_times) > 1000:
            self._execution_times = self._execution_times[-1000:]
        
        return combined_signal, lookback_used, volatility
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for entire price history (for backtesting).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional signal columns
        """
        # Start timing for latency profiling
        start_time = time.perf_counter()
        
        result = df.copy()
        result['momentum_signal'] = 0.0
        result['reversion_signal'] = 0.0
        result['signal'] = 0.0
        result['volatility'] = 0.0
        
        min_bars = max(
            self.momentum_lookback, 
            self.low_vol_lookback, 
            self.mean_reversion_lookback
        ) + 20
        
        if len(df) < min_bars:
            return result
        
        closes = df['close']
        
        # Vectorized momentum signal
        rolling_mean = closes.rolling(window=self.momentum_lookback).mean()
        momentum = (closes - rolling_mean) / rolling_mean
        momentum_signal = (momentum * 10).clip(-1.0, 1.0)
        
        # Vectorized mean reversion signal (Bollinger Bands)
        sma = closes.rolling(window=self.mean_reversion_lookback).mean()
        std = closes.rolling(window=self.mean_reversion_lookback).std()
        
        upper_band = sma + (std * self.bb_std_dev_multiplier)
        lower_band = sma - (std * self.bb_std_dev_multiplier)
        
        # Z-score calculation
        z_score = (closes - sma) / (std * self.bb_std_dev_multiplier)
        
        # Mean reversion signal: invert z-score, stronger at extremes
        reversion_signal = -z_score.clip(-1.0, 1.0)
        
        # Apply weights
        total_weight = self.momentum_weight + self.mean_reversion_weight
        if total_weight > 0:
            combined = (
                (momentum_signal.fillna(0) * self.momentum_weight) +
                (reversion_signal.fillna(0) * self.mean_reversion_weight)
            ) / total_weight
        else:
            combined = pd.Series(0.0, index=df.index)
        
        result['momentum_signal'] = momentum_signal.fillna(0)
        result['reversion_signal'] = reversion_signal.fillna(0)
        result['signal'] = combined.clip(-1.0, 1.0)
        
        # Rolling volatility
        log_returns = np.log(closes / closes.shift(1))
        result['volatility'] = log_returns.rolling(window=20).std()
        
        # Record execution time (total time divided by number of bars)
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000.0
        per_bar_time_ms = total_time_ms / len(df) if len(df) > 0 else 0.0
        self._last_execution_time_ms = per_bar_time_ms
        self._execution_times.append(per_bar_time_ms)
        
        # Keep only last 1000 measurements
        if len(self._execution_times) > 1000:
            self._execution_times = self._execution_times[-1000:]
        
        return result
    
    def get_trade_signal(self, signal: float) -> str:
        """
        Convert numerical signal to trade action.
        
        Args:
            signal: Signal value [-1, 1]
            
        Returns:
            'BUY', 'SELL', 'CLOSE', or 'HOLD'
        """
        if signal > self.signal_threshold:
            return 'BUY'
        elif signal < -self.signal_threshold:
            return 'SELL'
        elif abs(signal) < self.neutral_threshold:
            return 'CLOSE'
        else:
            return 'HOLD'
    
    def get_parameters(self) -> dict:
        """Get current strategy parameters."""
        return {
            'momentum_lookback': self.momentum_lookback,
            'trend_lookback': self.trend_lookback,
            'signal_threshold': self.signal_threshold,
            'neutral_threshold': self.neutral_threshold,
            'momentum_weight': self.momentum_weight,
            'trend_weight': self.trend_weight,
            'adaptive_enabled': self.adaptive_enabled,
            'high_vol_lookback': self.high_vol_lookback,
            'low_vol_lookback': self.low_vol_lookback,
            'vol_multiplier': self.vol_multiplier,
            'mean_reversion_enabled': self.mean_reversion_enabled,
            'mean_reversion_weight': self.mean_reversion_weight,
            'mean_reversion_lookback': self.mean_reversion_lookback,
            'bb_std_dev_multiplier': self.bb_std_dev_multiplier,
            'reversion_threshold': self.reversion_threshold
        }


# Backward compatibility alias
MomentumStrategy = HybridStrategy
