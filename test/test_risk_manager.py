"""
Test Risk Manager
=================
Unit tests for risk management and kill switch functionality.

Tests cover:
1. Volatility-targeted position sizing
2. Daily loss limit checks
3. Stop-loss triggers
4. Kill switch conditions

Author: OTREP-X Development Team
Date: December 2025
"""

import pytest
from datetime import date, datetime
from unittest.mock import MagicMock

# Import modules under test
import sys
sys.path.insert(0, '.')

from risk_manager import RiskManager
from risk.kill_switch import (
    KillSwitch, 
    KillSwitchConfig, 
    KillSwitchReason,
    KillSwitchStatus
)


class TestRiskManagerVolatilityTargeting:
    """Tests for ATR-based volatility targeting position sizing."""
    
    def test_calculate_position_size_basic(self):
        """Test basic position sizing with ATR."""
        rm = RiskManager(
            stop_loss_pct=0.02,
            max_positions=3,
            daily_loss_limit_pct=0.01,
            target_risk_per_trade_usd=125.0  # CTO: $1000/8 = $125
        )
        
        # Entry at $100, ATR = $2 (2% volatility)
        entry_price = 100.0
        symbol_atr = 2.0
        
        size = rm.calculate_position_size(entry_price, symbol_atr)
        
        # Expected: risk_per_trade / (ATR * atr_multiplier)
        # $125 / ($2 * 1.5) = 41.67 shares
        assert size > 0
        assert isinstance(size, int)
    
    def test_calculate_position_size_high_volatility(self):
        """Test position sizing reduces with high volatility."""
        rm = RiskManager(
            target_risk_per_trade_usd=125.0
        )
        
        entry_price = 100.0
        
        # Low volatility
        size_low_vol = rm.calculate_position_size(entry_price, symbol_atr=1.0)
        
        # High volatility
        size_high_vol = rm.calculate_position_size(entry_price, symbol_atr=5.0)
        
        # Higher volatility should result in smaller position
        assert size_low_vol > size_high_vol
    
    def test_calculate_position_size_minimum(self):
        """Test position size has a minimum of 1 share."""
        rm = RiskManager(
            target_risk_per_trade_usd=10.0  # Very small risk budget
        )
        
        # Very high volatility
        size = rm.calculate_position_size(entry_price=100.0, symbol_atr=100.0)
        
        # Should still be at least 1
        assert size >= 1
    
    def test_calculate_position_size_maximum(self):
        """Test position size has a maximum cap."""
        rm = RiskManager(
            target_risk_per_trade_usd=10000.0,  # Large risk budget
            max_position_size_usd=5000.0
        )
        
        entry_price = 10.0  # Cheap stock
        symbol_atr = 0.1    # Very low volatility
        
        size = rm.calculate_position_size(entry_price, symbol_atr)
        
        # Should not exceed max_position_size_usd / entry_price
        assert size <= rm.max_position_size_usd / entry_price


class TestRiskManagerDailyLoss:
    """Tests for daily loss limit functionality."""
    
    def test_daily_loss_limit_not_breached(self):
        """Test trading continues when under limit."""
        rm = RiskManager(daily_loss_limit_pct=0.01)  # 1%
        rm.set_starting_equity(100000.0)
        
        # Equity down 0.5% (under limit)
        result = rm.check_daily_loss_limit(99500.0)
        
        assert result is True
        assert rm.trading_halted is False
    
    def test_daily_loss_limit_breached(self):
        """Test trading halts when limit breached."""
        rm = RiskManager(daily_loss_limit_pct=0.01)  # 1%
        rm.set_starting_equity(100000.0)
        
        # Equity down 1.5% (over limit)
        result = rm.check_daily_loss_limit(98500.0)
        
        assert result is False
        assert rm.trading_halted is True
    
    def test_daily_loss_limit_exactly_at_limit(self):
        """Test behavior at exact limit threshold."""
        rm = RiskManager(daily_loss_limit_pct=0.01)
        rm.set_starting_equity(100000.0)
        
        # Exactly at 1% loss
        result = rm.check_daily_loss_limit(99000.0)
        
        # Should halt (<=)
        assert result is False


class TestRiskManagerStopLoss:
    """Tests for stop-loss functionality."""
    
    def test_stop_loss_not_triggered(self):
        """Test stop-loss not triggered when above threshold."""
        rm = RiskManager(stop_loss_pct=0.02)  # 2%
        
        result = rm.check_stop_loss(
            symbol='AAPL',
            entry_price=100.0,
            current_price=99.0  # 1% down, under 2% limit
        )
        
        assert result is False
    
    def test_stop_loss_triggered(self):
        """Test stop-loss triggers when below threshold."""
        rm = RiskManager(stop_loss_pct=0.02)
        
        result = rm.check_stop_loss(
            symbol='AAPL',
            entry_price=100.0,
            current_price=97.0  # 3% down, over 2% limit
        )
        
        assert result is True
    
    def test_stop_loss_exactly_at_limit(self):
        """Test stop-loss at exact threshold."""
        rm = RiskManager(stop_loss_pct=0.02)
        
        result = rm.check_stop_loss(
            symbol='AAPL',
            entry_price=100.0,
            current_price=98.0  # Exactly 2% down
        )
        
        # Should trigger (<=)
        assert result is True


class TestKillSwitch:
    """Tests for the centralized kill switch."""
    
    def test_kill_switch_initial_state(self):
        """Test kill switch starts in non-triggered state."""
        ks = KillSwitch()
        
        assert ks.is_trading_allowed() is True
        assert ks.get_status().is_triggered is False
    
    def test_kill_switch_daily_drawdown_trigger(self):
        """Test kill switch triggers on daily drawdown."""
        config = KillSwitchConfig(daily_drawdown_limit_pct=0.004)  # 0.40%
        ks = KillSwitch(config=config)
        ks.set_starting_equity(250000.0)  # $250k
        
        # Drawdown of 0.5% (exceeds 0.4%)
        current_equity = 250000 * (1 - 0.005)
        status = ks.trigger_check(current_equity=current_equity)
        
        assert status.is_triggered is True
        assert status.reason == KillSwitchReason.DAILY_DRAWDOWN
        assert ks.is_trading_allowed() is False
    
    def test_kill_switch_drawdown_under_limit(self):
        """Test kill switch does not trigger under limit."""
        config = KillSwitchConfig(daily_drawdown_limit_pct=0.004)
        ks = KillSwitch(config=config)
        ks.set_starting_equity(250000.0)
        
        # Drawdown of 0.3% (under 0.4%)
        current_equity = 250000 * (1 - 0.003)
        status = ks.trigger_check(current_equity=current_equity)
        
        assert status.is_triggered is False
        assert ks.is_trading_allowed() is True
    
    def test_kill_switch_order_rejects(self):
        """Test kill switch triggers on excessive order rejects."""
        config = KillSwitchConfig(max_order_rejects=5)
        ks = KillSwitch(config=config)
        
        # Record 6 consecutive rejects
        for _ in range(6):
            ks.record_order_result(success=False)
        
        status = ks.trigger_check()
        
        assert status.is_triggered is True
        assert status.reason == KillSwitchReason.ORDER_REJECTS
    
    def test_kill_switch_order_reject_reset(self):
        """Test order reject counter resets on success."""
        config = KillSwitchConfig(max_order_rejects=5)
        ks = KillSwitch(config=config)
        
        # Record 4 rejects
        for _ in range(4):
            ks.record_order_result(success=False)
        
        # One success resets counter
        ks.record_order_result(success=True)
        
        # 3 more rejects (total 3, under limit)
        for _ in range(3):
            ks.record_order_result(success=False)
        
        status = ks.trigger_check()
        
        assert status.is_triggered is False
    
    def test_kill_switch_manual_trigger(self):
        """Test manual kill switch trigger."""
        ks = KillSwitch()
        
        ks.manual_trigger("Emergency shutdown")
        
        assert ks.is_trading_allowed() is False
        assert ks.get_status().reason == KillSwitchReason.MANUAL
    
    def test_kill_switch_manual_reset(self):
        """Test manual reset of kill switch."""
        ks = KillSwitch()
        ks.manual_trigger("Test")
        
        assert ks.is_trading_allowed() is False
        
        ks.manual_reset()
        
        assert ks.is_trading_allowed() is True
    
    def test_kill_switch_metrics(self):
        """Test metrics collection."""
        ks = KillSwitch()
        ks.set_starting_equity(100000.0)
        
        ks.record_order_result(success=True)
        ks.record_order_result(success=False)
        ks.record_data_update()
        
        metrics = ks.get_metrics()
        
        assert 'is_triggered' in metrics
        assert 'consecutive_order_rejects' in metrics
        assert 'starting_equity' in metrics
        assert metrics['consecutive_order_rejects'] == 1


class TestKillSwitchConfig:
    """Tests for kill switch configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = KillSwitchConfig()
        
        assert config.daily_drawdown_limit_pct == 0.004  # 0.40%
        assert config.max_order_rejects == 5
        assert config.max_data_staleness_seconds == 600.0  # 10 min
        assert config.max_api_failures == 3
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = KillSwitchConfig(
            daily_drawdown_limit_pct=0.01,
            max_order_rejects=10
        )
        
        assert config.daily_drawdown_limit_pct == 0.01
        assert config.max_order_rejects == 10


# Entry point for pytest
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
