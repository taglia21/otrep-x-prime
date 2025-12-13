"""
Test Position Sizer
===================
Unit tests for ATR-based volatility targeting position sizing.

Author: OTREP-X Development Team
Date: December 2025
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, '.')

from risk.position_sizer import (
    PositionSizer,
    PositionSizerConfig,
    PositionSizeResult,
    compute_atr,
    compute_position_size,
)


class TestATRCalculation:
    """Tests for ATR (Average True Range) calculation."""
    
    def test_atr_basic(self):
        """Test basic ATR calculation."""
        sizer = PositionSizer()
        
        # Simple case: constant high-low range of $2
        highs = np.array([102, 104, 103, 105, 104] * 3)  # 15 bars
        lows = np.array([100, 102, 101, 103, 102] * 3)
        closes = np.array([101, 103, 102, 104, 103] * 3)
        
        atr = sizer.calculate_atr(highs, lows, closes, period=14)
        
        # Should be approximately 2 (high - low)
        assert atr > 0
        assert abs(atr - 2.0) < 1.0  # Within $1 of expected
    
    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data falls back to simple range."""
        sizer = PositionSizer()
        
        # Only 5 bars (less than period + 1)
        highs = np.array([102, 104, 103, 105, 104])
        lows = np.array([100, 102, 101, 103, 102])
        closes = np.array([101, 103, 102, 104, 103])
        
        atr = sizer.calculate_atr(highs, lows, closes, period=14)
        
        # Should return mean of high-low
        assert atr > 0
    
    def test_atr_empty_data(self):
        """Test ATR with empty data returns 0."""
        sizer = PositionSizer()
        
        atr = sizer.calculate_atr(np.array([]), np.array([]), np.array([]))
        
        assert atr == 0.0
    
    def test_atr_from_bars(self):
        """Test ATR calculation from bar dictionaries."""
        sizer = PositionSizer()
        
        bars = [
            {'high': 102, 'low': 100, 'close': 101},
            {'high': 104, 'low': 102, 'close': 103},
            {'high': 103, 'low': 101, 'close': 102},
        ] * 5  # 15 bars
        
        atr = sizer.calculate_atr_from_bars(bars, period=14)
        
        assert atr > 0
    
    def test_standalone_atr(self):
        """Test standalone compute_atr function."""
        highs = np.array([102, 104, 103, 105, 104] * 3)
        lows = np.array([100, 102, 101, 103, 102] * 3)
        closes = np.array([101, 103, 102, 104, 103] * 3)
        
        atr = compute_atr(highs, lows, closes, period=14)
        
        assert atr > 0


class TestPositionSizing:
    """Tests for position sizing calculations."""
    
    def test_basic_sizing(self):
        """Test basic position sizing with ATR."""
        config = PositionSizerConfig(
            target_risk_per_trade_usd=125.0,
            atr_multiplier=1.5
        )
        sizer = PositionSizer(config)
        
        result = sizer.calculate_size(
            entry_price=100.0,
            atr=2.0,  # $2 ATR
            symbol='AAPL'
        )
        
        # risk_per_share = 2.0 * 1.5 = $3
        # shares = 125 / 3 = 41.67 -> 41 shares
        assert isinstance(result, PositionSizeResult)
        assert result.shares > 0
        assert result.shares == 41
        assert result.notional_usd == 4100.0  # 41 * 100
    
    def test_sizing_minimum(self):
        """Test minimum position size of 1 share."""
        config = PositionSizerConfig(
            target_risk_per_trade_usd=10.0,  # Very small budget
            atr_multiplier=1.5
        )
        sizer = PositionSizer(config)
        
        result = sizer.calculate_size(
            entry_price=100.0,
            atr=50.0  # Very high volatility
        )
        
        assert result.shares >= 1
    
    def test_sizing_maximum_cap(self):
        """Test position size is capped at maximum."""
        config = PositionSizerConfig(
            target_risk_per_trade_usd=10000.0,  # Large budget
            max_position_size_usd=5000.0,  # Cap at $5k
            atr_multiplier=1.5
        )
        sizer = PositionSizer(config)
        
        result = sizer.calculate_size(
            entry_price=10.0,  # Cheap stock
            atr=0.1  # Low volatility
        )
        
        # Should be capped at $5000 / $10 = 500 shares
        assert result.shares <= 500
        assert result.notional_usd <= 5000.0
        assert result.was_capped is True
        assert result.cap_reason == 'max_position_size'
    
    def test_sizing_zero_atr_uses_proxy(self):
        """Test position sizing uses 2% proxy when ATR is 0."""
        config = PositionSizerConfig(
            target_risk_per_trade_usd=125.0,
            atr_multiplier=1.5
        )
        sizer = PositionSizer(config)
        
        result = sizer.calculate_size(
            entry_price=100.0,
            atr=0.0  # No ATR available
        )
        
        # Should use 2% of price = $2 as ATR proxy
        # risk_per_share = 2.0 * 1.5 = $3
        # shares = 125 / 3 = 41
        assert result.shares > 0
        assert result.atr == 2.0  # 2% of $100
    
    def test_sizing_invalid_price(self):
        """Test position sizing with invalid price returns minimum."""
        sizer = PositionSizer()
        
        result = sizer.calculate_size(entry_price=0.0, atr=2.0)
        
        assert result.shares == 1
        assert result.was_capped is True
        assert result.cap_reason == 'invalid_entry_price'
    
    def test_standalone_compute_position_size(self):
        """Test standalone compute_position_size function."""
        shares = compute_position_size(
            entry_price=100.0,
            atr=2.0,
            target_risk_usd=125.0,
            atr_multiplier=1.5
        )
        
        assert shares == 41
    
    def test_risk_budget_calculation(self):
        """Test risk budget calculation from daily limit."""
        config = PositionSizerConfig(
            daily_loss_limit_usd=1000.0,
            expected_stop_outs=8
        )
        sizer = PositionSizer(config)
        
        risk_per_trade = sizer.calculate_risk_budget()
        
        assert risk_per_trade == 125.0  # $1000 / 8
    
    def test_risk_budget_floor(self):
        """Test risk budget has $25 floor."""
        config = PositionSizerConfig(
            daily_loss_limit_usd=100.0,  # Small daily budget
            expected_stop_outs=20  # Many expected stop-outs
        )
        sizer = PositionSizer(config)
        
        # Would be $5 without floor
        risk_per_trade = sizer.calculate_risk_budget()
        
        assert risk_per_trade == 25.0  # Floored at $25
    
    def test_batch_calculate(self):
        """Test batch position size calculation."""
        sizer = PositionSizer()
        
        positions = [
            {'symbol': 'AAPL', 'entry_price': 150.0},
            {'symbol': 'MSFT', 'entry_price': 400.0},
            {'symbol': 'GOOGL', 'entry_price': 180.0},
        ]
        
        atr_data = {
            'AAPL': 3.0,
            'MSFT': 5.0,
            'GOOGL': 4.0,
        }
        
        results = sizer.batch_calculate(positions, atr_data)
        
        assert 'AAPL' in results
        assert 'MSFT' in results
        assert 'GOOGL' in results
        
        # MSFT is more expensive, should have fewer shares
        assert results['AAPL'].shares >= results['MSFT'].shares


class TestPositionSizerConfig:
    """Tests for position sizer configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PositionSizerConfig()
        
        assert config.target_risk_per_trade_usd == 125.0
        assert config.max_position_size_usd == 5000.0
        assert config.atr_multiplier == 1.5
        assert config.atr_period == 14
        assert config.min_shares == 1
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PositionSizerConfig(
            target_risk_per_trade_usd=200.0,
            max_position_size_usd=10000.0,
            atr_multiplier=2.0
        )
        
        assert config.target_risk_per_trade_usd == 200.0
        assert config.max_position_size_usd == 10000.0
        assert config.atr_multiplier == 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
