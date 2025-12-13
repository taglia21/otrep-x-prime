"""
Kill Switch - Centralized Circuit Breaker
==========================================
Global emergency shutdown mechanism based on CTO's hardening plan.

Trigger Conditions:
1. Daily Drawdown Limit: 0.40% of starting equity
2. Order Reject Count: > 5 consecutive rejects
3. Data Staleness: > 10 minutes since last update

Author: OTREP-X Development Team
Date: December 2025
"""

import time
from datetime import datetime, date
from enum import Enum
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field


class KillSwitchReason(Enum):
    """Enumeration of kill switch trigger reasons."""
    NONE = "none"
    DAILY_DRAWDOWN = "daily_drawdown_exceeded"
    ORDER_REJECTS = "excessive_order_rejects"
    DATA_STALENESS = "stale_market_data"
    MANUAL = "manual_shutdown"
    API_FAILURE = "api_connection_failure"
    POSITION_LIMIT_BREACH = "position_limit_breach"


@dataclass
class KillSwitchStatus:
    """Current status of the kill switch."""
    is_triggered: bool = False
    reason: KillSwitchReason = KillSwitchReason.NONE
    triggered_at: Optional[datetime] = None
    message: str = ""
    
    # Metrics at time of trigger
    drawdown_pct: float = 0.0
    order_reject_count: int = 0
    data_staleness_seconds: float = 0.0


@dataclass
class KillSwitchConfig:
    """Configuration for kill switch thresholds."""
    # CTO Directive: 0.40% daily drawdown limit
    # Rationale: $1000 daily budget / $250k portfolio = 0.4%
    daily_drawdown_limit_pct: float = 0.004  # 0.40%
    
    # Order reject threshold
    max_order_rejects: int = 5
    
    # Data staleness threshold (seconds)
    # 10 minutes = 600 seconds
    max_data_staleness_seconds: float = 600.0
    
    # API failure threshold (consecutive failures)
    max_api_failures: int = 3


class KillSwitch:
    """
    Centralized Circuit Breaker for OTREP-X PRIME.
    
    Implements multiple kill conditions as specified by CTO:
    1. Daily drawdown limit (0.40%)
    2. Order reject count (>5)
    3. Data staleness (>10 min)
    4. API connection failures (>3)
    
    Usage:
        kill_switch = KillSwitch(config, logger=print)
        
        # Before any trade attempt:
        if not kill_switch.is_trading_allowed():
            return  # Trading halted
        
        # Update metrics after each operation:
        kill_switch.record_order_result(success=True)
        kill_switch.record_data_update()
        
        # Check trigger status:
        kill_switch.trigger_check(current_equity, starting_equity)
    """
    
    def __init__(
        self,
        config: Optional[KillSwitchConfig] = None,
        logger: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize kill switch with configuration.
        
        Args:
            config: Kill switch configuration (uses defaults if None)
            logger: Logging function (defaults to print)
        """
        self.config = config or KillSwitchConfig()
        self.log = logger or print
        
        # State tracking
        self._status = KillSwitchStatus()
        self._starting_equity: float = 0.0
        self._halt_date: Optional[date] = None
        
        # Counters
        self._consecutive_order_rejects: int = 0
        self._consecutive_api_failures: int = 0
        self._last_data_update: float = time.time()
        
        # Position tracking
        self._current_position_count: int = 0
        self._max_positions: int = 3
    
    def set_starting_equity(self, equity: float) -> None:
        """Set starting equity for daily drawdown calculation."""
        self._starting_equity = equity
        self.log(f"🎯 KillSwitch: Starting equity = ${equity:,.2f}")
        self.log(f"   Daily drawdown limit: {self.config.daily_drawdown_limit_pct:.2%} "
                f"(${equity * self.config.daily_drawdown_limit_pct:,.2f})")
    
    def set_max_positions(self, max_positions: int) -> None:
        """Set maximum position limit."""
        self._max_positions = max_positions
    
    def is_trading_allowed(self) -> bool:
        """
        Check if trading is currently allowed.
        
        Returns:
            True if trading is allowed, False if kill switch is triggered
        """
        # Reset if new trading day
        today = date.today()
        if self._halt_date and self._halt_date < today:
            self._reset_for_new_day()
        
        return not self._status.is_triggered
    
    def get_status(self) -> KillSwitchStatus:
        """Get current kill switch status."""
        return self._status
    
    def trigger_check(
        self,
        current_equity: Optional[float] = None,
        starting_equity: Optional[float] = None
    ) -> KillSwitchStatus:
        """
        Perform comprehensive trigger check.
        
        This is the main method to call before any trading action.
        Checks all kill conditions and updates status.
        
        Args:
            current_equity: Current portfolio equity (optional)
            starting_equity: Starting equity for the day (optional, uses cached)
            
        Returns:
            Current KillSwitchStatus
        """
        if self._status.is_triggered:
            return self._status
        
        # Update starting equity if provided
        if starting_equity is not None:
            self._starting_equity = starting_equity
        
        # Check 1: Daily Drawdown
        if current_equity is not None and self._starting_equity > 0:
            drawdown = (current_equity - self._starting_equity) / self._starting_equity
            if drawdown <= -self.config.daily_drawdown_limit_pct:
                self._trigger(
                    reason=KillSwitchReason.DAILY_DRAWDOWN,
                    message=f"Daily drawdown {drawdown:.3%} exceeds limit {-self.config.daily_drawdown_limit_pct:.3%}",
                    drawdown_pct=drawdown
                )
                return self._status
        
        # Check 2: Order Rejects
        if self._consecutive_order_rejects > self.config.max_order_rejects:
            self._trigger(
                reason=KillSwitchReason.ORDER_REJECTS,
                message=f"Consecutive order rejects ({self._consecutive_order_rejects}) "
                       f"exceeds limit ({self.config.max_order_rejects})",
                order_reject_count=self._consecutive_order_rejects
            )
            return self._status
        
        # Check 3: Data Staleness
        staleness = time.time() - self._last_data_update
        if staleness > self.config.max_data_staleness_seconds:
            self._trigger(
                reason=KillSwitchReason.DATA_STALENESS,
                message=f"Data staleness ({staleness:.0f}s) exceeds limit "
                       f"({self.config.max_data_staleness_seconds:.0f}s)",
                data_staleness_seconds=staleness
            )
            return self._status
        
        # Check 4: API Failures
        if self._consecutive_api_failures > self.config.max_api_failures:
            self._trigger(
                reason=KillSwitchReason.API_FAILURE,
                message=f"Consecutive API failures ({self._consecutive_api_failures}) "
                       f"exceeds limit ({self.config.max_api_failures})",
            )
            return self._status
        
        return self._status
    
    def record_order_result(self, success: bool) -> None:
        """
        Record order submission result.
        
        Args:
            success: True if order was accepted, False if rejected
        """
        if success:
            self._consecutive_order_rejects = 0
        else:
            self._consecutive_order_rejects += 1
            self.log(f"⚠️ Order reject #{self._consecutive_order_rejects}")
            
            if self._consecutive_order_rejects > self.config.max_order_rejects:
                self.trigger_check()
    
    def record_data_update(self) -> None:
        """Record successful data update (resets staleness timer)."""
        self._last_data_update = time.time()
    
    def record_api_result(self, success: bool) -> None:
        """
        Record API call result.
        
        Args:
            success: True if API call succeeded, False if failed
        """
        if success:
            self._consecutive_api_failures = 0
        else:
            self._consecutive_api_failures += 1
            self.log(f"⚠️ API failure #{self._consecutive_api_failures}")
            
            if self._consecutive_api_failures > self.config.max_api_failures:
                self.trigger_check()
    
    def update_position_count(self, count: int) -> None:
        """Update current position count."""
        self._current_position_count = count
    
    def manual_trigger(self, reason: str = "Manual shutdown requested") -> None:
        """Manually trigger the kill switch."""
        self._trigger(
            reason=KillSwitchReason.MANUAL,
            message=reason
        )
    
    def manual_reset(self) -> None:
        """Manually reset the kill switch (use with caution)."""
        self.log("⚠️ MANUAL KILL SWITCH RESET")
        self._reset_for_new_day()
    
    def _trigger(
        self,
        reason: KillSwitchReason,
        message: str,
        drawdown_pct: float = 0.0,
        order_reject_count: int = 0,
        data_staleness_seconds: float = 0.0
    ) -> None:
        """Internal method to trigger the kill switch."""
        self._status = KillSwitchStatus(
            is_triggered=True,
            reason=reason,
            triggered_at=datetime.now(),
            message=message,
            drawdown_pct=drawdown_pct,
            order_reject_count=order_reject_count or self._consecutive_order_rejects,
            data_staleness_seconds=data_staleness_seconds
        )
        self._halt_date = date.today()
        
        # Log the trigger with high visibility
        self.log("")
        self.log("=" * 70)
        self.log("🚨🚨🚨 KILL SWITCH TRIGGERED 🚨🚨🚨")
        self.log("=" * 70)
        self.log(f"   Reason: {reason.value}")
        self.log(f"   Message: {message}")
        self.log(f"   Time: {self._status.triggered_at}")
        self.log("   ALL TRADING HALTED UNTIL MANUAL RESET OR NEW DAY")
        self.log("=" * 70)
        self.log("")
    
    def _reset_for_new_day(self) -> None:
        """Reset kill switch state for a new trading day."""
        self.log("🔄 KillSwitch: New trading day - resetting state")
        self._status = KillSwitchStatus()
        self._halt_date = None
        self._consecutive_order_rejects = 0
        self._consecutive_api_failures = 0
        self._last_data_update = time.time()
    
    def get_metrics(self) -> Dict:
        """Get current metrics for monitoring."""
        staleness = time.time() - self._last_data_update
        return {
            'is_triggered': self._status.is_triggered,
            'reason': self._status.reason.value,
            'triggered_at': self._status.triggered_at.isoformat() if self._status.triggered_at else None,
            'starting_equity': self._starting_equity,
            'daily_drawdown_limit_pct': self.config.daily_drawdown_limit_pct,
            'consecutive_order_rejects': self._consecutive_order_rejects,
            'max_order_rejects': self.config.max_order_rejects,
            'data_staleness_seconds': staleness,
            'max_data_staleness_seconds': self.config.max_data_staleness_seconds,
            'consecutive_api_failures': self._consecutive_api_failures,
            'max_api_failures': self.config.max_api_failures
        }


# Convenience function for configuration from YAML
def create_kill_switch_from_config(config: Dict, logger=None) -> KillSwitch:
    """
    Create KillSwitch from YAML config dict.
    
    Args:
        config: Dictionary with RISK section from config.yaml
        logger: Optional logging function
        
    Returns:
        Configured KillSwitch instance
    """
    risk_config = config.get('RISK', {})
    
    ks_config = KillSwitchConfig(
        # CTO directive: 0.40% daily drawdown
        daily_drawdown_limit_pct=risk_config.get('KILL_SWITCH_DRAWDOWN_PCT', 0.004),
        max_order_rejects=risk_config.get('KILL_SWITCH_MAX_REJECTS', 5),
        max_data_staleness_seconds=risk_config.get('KILL_SWITCH_DATA_STALE_SEC', 600),
        max_api_failures=risk_config.get('KILL_SWITCH_MAX_API_FAILS', 3)
    )
    
    return KillSwitch(config=ks_config, logger=logger)
