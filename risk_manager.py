"""
Risk Manager
============
Essential risk management controls for the trading system.

Features:
- Volatility-targeted position sizing (ATR-based)
- Stop-loss monitoring
- Max position limit
- Daily loss limit (circuit breaker)

CTO Directive (Dec 2025):
- Replace fixed $2,000 sizing with volatility targeting
- Target: $1000 daily budget / 8 expected stop-outs = $125 max risk per trade

Author: OTREP-X Development Team
Lead Engineer: Gemini AI
"""

from datetime import date
from typing import Optional, Callable


class RiskManager:
    """
    Essential risk management controls.
    
    Implements:
    - Volatility-targeted position sizing (ATR-based)
    - Stop-loss monitoring on entry price
    - Max concurrent position limit
    - Daily loss limit circuit breaker
    """
    
    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        max_positions: int = 3,
        daily_loss_limit_pct: float = 0.01,
        target_risk_per_trade_usd: float = 125.0,
        max_position_size_usd: float = 5000.0,
        atr_risk_multiplier: float = 1.5,
        logger: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize risk manager with limits.
        
        Args:
            stop_loss_pct: Stop-loss percentage (0.02 = 2%)
            max_positions: Maximum concurrent positions allowed
            daily_loss_limit_pct: Daily drawdown limit (0.01 = 1%)
            target_risk_per_trade_usd: Target dollar risk per trade ($125 default)
            max_position_size_usd: Maximum position size cap ($5000 default)
            atr_risk_multiplier: ATR multiplier for risk calculation (1.5 default)
            logger: Optional logging function
        """
        self.stop_loss_pct = stop_loss_pct
        self.max_positions = max_positions
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.target_risk_per_trade_usd = target_risk_per_trade_usd
        self.max_position_size_usd = max_position_size_usd
        self.atr_risk_multiplier = atr_risk_multiplier
        self.log = logger or print
        
        # State tracking
        self.starting_equity: float = 0.0
        self.trading_halted: bool = False
        self.halt_date: Optional[date] = None
    
    def calculate_position_size(
        self,
        entry_price: float,
        symbol_atr: float,
        override_risk_usd: Optional[float] = None
    ) -> int:
        """
        Calculate position size using volatility targeting.
        
        Uses ATR (Average True Range) to determine position size that
        achieves a consistent dollar risk per trade.
        
        Formula: size = target_risk / (ATR * multiplier)
        
        CTO Rationale:
        - Daily budget: $1,000
        - Expected stop-outs: 8 per day
        - Target risk per trade: $1,000 / 8 = $125
        
        Args:
            entry_price: Expected entry price
            symbol_atr: 14-period ATR for the symbol
            override_risk_usd: Override target risk (optional)
            
        Returns:
            Number of shares to trade (integer, minimum 1)
        """
        if entry_price <= 0 or symbol_atr <= 0:
            self.log("⚠️ Invalid entry_price or ATR, using minimum size")
            return 1
        
        target_risk = override_risk_usd or self.target_risk_per_trade_usd
        
        # Calculate dollar risk per share
        risk_per_share = symbol_atr * self.atr_risk_multiplier
        
        # Calculate raw position size
        raw_size = target_risk / risk_per_share
        
        # Apply maximum position size cap
        max_shares = self.max_position_size_usd / entry_price
        capped_size = min(raw_size, max_shares)
        
        # Ensure minimum of 1 share
        final_size = max(1, int(capped_size))
        
        # Log the calculation
        notional = final_size * entry_price
        expected_risk = final_size * risk_per_share
        self.log(f"📐 Position sizing: {final_size} shares @ ${entry_price:.2f} = ${notional:.2f}")
        self.log(f"   ATR: ${symbol_atr:.2f}, Risk/share: ${risk_per_share:.2f}, Expected risk: ${expected_risk:.2f}")
        
        return final_size
    
    def set_starting_equity(self, equity: float) -> None:
        """
        Set starting equity for daily P&L tracking.
        
        Should be called at start of each trading day.
        
        Args:
            equity: Starting portfolio equity value
        """
        self.starting_equity = equity
        self.log(f"📊 Starting equity set: ${equity:,.2f}")
    
    def check_daily_loss_limit(self, current_equity: float) -> bool:
        """
        Check if daily loss limit has been breached.
        
        Implements circuit breaker pattern - halts all trading
        if daily drawdown exceeds limit.
        
        Args:
            current_equity: Current portfolio equity value
            
        Returns:
            True if trading should continue, False if halted
        """
        # Reset halt if it's a new day
        today = date.today()
        if self.halt_date and self.halt_date < today:
            self.trading_halted = False
            self.halt_date = None
            self.log("🔄 New trading day - halt lifted")
        
        if self.trading_halted:
            return False
        
        if self.starting_equity <= 0:
            return True
        
        daily_drawdown = (current_equity - self.starting_equity) / self.starting_equity
        
        if daily_drawdown <= -self.daily_loss_limit_pct:
            self.trading_halted = True
            self.halt_date = today
            self.log(f"🛑 DAILY LOSS LIMIT HIT: {daily_drawdown:.2%}")
            self.log("🛑 Trading halted until next market open.")
            return False
        
        return True
    
    def check_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        current_price: float
    ) -> bool:
        """
        Check if stop-loss has been triggered.
        
        Args:
            symbol: Stock ticker symbol
            entry_price: Average entry price for position
            current_price: Current market price
            
        Returns:
            True if stop-loss triggered (should close position)
        """
        if entry_price <= 0:
            return False
        
        pl_pct = (current_price - entry_price) / entry_price
        
        if pl_pct <= -self.stop_loss_pct:
            self.log(f"🚨 STOP LOSS TRIGGERED: {symbol}")
            self.log(f"   Entry: ${entry_price:.2f} → Current: ${current_price:.2f}")
            self.log(f"   Loss: {pl_pct:.2%} (limit: {-self.stop_loss_pct:.2%})")
            return True
        
        return False
    
    def check_max_positions(self, current_position_count: int) -> bool:
        """
        DEPRECATED: Use can_open_position() instead.
        Check if we can open a new position.
        
        Args:
            current_position_count: Number of currently open positions
            
        Returns:
            True if new position allowed, False if at limit
        """
        return self.can_open_position(current_position_count)
    
    def can_open_position(self, current_open_count: int) -> bool:
        """
        Check if we can open a new position (transactional check).
        
        This method uses the synchronized position count from the broker
        to ensure accurate position limit enforcement.
        
        Args:
            current_open_count: Current number of open positions from broker sync
            
        Returns:
            True if new position allowed, False if at limit
        """
        if self.trading_halted:
            return False
        return current_open_count < self.max_positions
    
    def get_status(self) -> dict:
        """
        Get current risk manager status.
        
        Returns:
            Dict with current limits and state
        """
        return {
            'stop_loss_pct': self.stop_loss_pct,
            'max_positions': self.max_positions,
            'daily_loss_limit_pct': self.daily_loss_limit_pct,
            'starting_equity': self.starting_equity,
            'trading_halted': self.trading_halted,
            'halt_date': self.halt_date.isoformat() if self.halt_date else None
        }
    
    def calculate_position_risk(
        self,
        entry_price: float,
        current_price: float,
        position_size: float,
        symbol: str = ''
    ) -> dict:
        """
        Calculate risk metrics for a position.
        
        Args:
            entry_price: Position entry price
            current_price: Current market price
            position_size: Position size in dollars
            symbol: Optional symbol for logging
            
        Returns:
            Dict with risk metrics including stop-loss trigger status
        """
        if entry_price <= 0:
            return {
                'stop_price': 0.0,
                'max_loss_usd': 0.0,
                'current_pl_pct': 0.0,
                'distance_to_stop_pct': 1.0,
                'is_stop_triggered': False
            }
        
        stop_price = entry_price * (1 - self.stop_loss_pct)
        max_loss = position_size * self.stop_loss_pct
        current_pl_pct = (current_price - entry_price) / entry_price
        distance_to_stop = (current_price - stop_price) / current_price
        is_stop_triggered = current_pl_pct <= -self.stop_loss_pct
        
        if is_stop_triggered and symbol:
            self.log(f"🚨 STOP LOSS TRIGGERED: {symbol}")
            self.log(f"   Entry: ${entry_price:.2f} → Current: ${current_price:.2f}")
            self.log(f"   Loss: {current_pl_pct:.2%} (limit: {-self.stop_loss_pct:.2%})")
        
        return {
            'stop_price': stop_price,
            'max_loss_usd': max_loss,
            'current_pl_pct': current_pl_pct,
            'distance_to_stop_pct': distance_to_stop,
            'is_stop_triggered': is_stop_triggered
        }
    
    # =========================================================================
    # PRE-TRADE / POST-TRADE INTERFACE (CTO Directive)
    # =========================================================================
    
    def pre_trade_check(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        shares: int,
        current_positions: int,
        current_equity: float
    ) -> dict:
        """
        Comprehensive pre-trade risk check.
        
        Runs all risk checks before submitting an order:
        1. Daily loss limit check
        2. Position count limit
        3. Trading halt check
        
        Args:
            symbol: Stock ticker
            side: 'BUY' or 'SELL'
            entry_price: Expected entry price
            shares: Number of shares
            current_positions: Current open position count
            current_equity: Current portfolio equity
            
        Returns:
            Dict with 'allowed' bool and 'reason' if blocked
        """
        notional = shares * entry_price
        
        # Check 1: Trading halt
        if self.trading_halted:
            return {
                'allowed': False,
                'reason': 'trading_halted',
                'message': 'Trading halted due to daily loss limit'
            }
        
        # Check 2: Daily loss limit
        if not self.check_daily_loss_limit(current_equity):
            return {
                'allowed': False,
                'reason': 'daily_loss_limit',
                'message': f'Daily loss limit exceeded'
            }
        
        # Check 3: Position count limit (for new positions)
        if side == 'BUY' and not self.can_open_position(current_positions):
            return {
                'allowed': False,
                'reason': 'max_positions',
                'message': f'Position limit reached ({self.max_positions})'
            }
        
        # Check 4: Position size limit
        if notional > self.max_position_size_usd:
            return {
                'allowed': False,
                'reason': 'position_size_limit',
                'message': f'Position ${notional:.2f} exceeds limit ${self.max_position_size_usd:.2f}'
            }
        
        # All checks passed
        self.log(f"✅ Pre-trade check passed: {side} {shares} {symbol} @ ${entry_price:.2f}")
        return {
            'allowed': True,
            'reason': None,
            'message': 'All risk checks passed',
            'notional_usd': notional
        }
    
    def post_trade_update(
        self,
        symbol: str,
        side: str,
        fill_price: float,
        shares: int,
        order_id: str = ''
    ) -> None:
        """
        Update risk state after a trade is executed.
        
        Called after an order is filled to update internal tracking.
        
        Args:
            symbol: Stock ticker
            side: 'BUY' or 'SELL'
            fill_price: Actual fill price
            shares: Number of shares filled
            order_id: Broker order ID
        """
        notional = shares * fill_price
        
        # Track realized P&L would go here if we had position tracking
        # For now, just log the trade
        self.log(f"📝 Post-trade: {side} {shares} {symbol} @ ${fill_price:.2f} = ${notional:.2f}")
        if order_id:
            self.log(f"   Order ID: {order_id}")
    
    # =========================================================================
    # P&L TRACKING
    # =========================================================================
    
    def calculate_unrealized_pnl(
        self,
        positions: list
    ) -> float:
        """
        Calculate total unrealized P&L from positions.
        
        Args:
            positions: List of dicts with 'entry_price', 'current_price', 'shares'
            
        Returns:
            Total unrealized P&L in dollars
        """
        total_pnl = 0.0
        
        for pos in positions:
            entry = pos.get('entry_price', 0)
            current = pos.get('current_price', 0)
            shares = pos.get('shares', 0)
            
            pnl = (current - entry) * shares
            total_pnl += pnl
        
        return total_pnl
    
    def calculate_current_equity(
        self,
        cash_balance: float,
        positions: list
    ) -> float:
        """
        Calculate current equity including unrealized P&L.
        
        Args:
            cash_balance: Current cash balance
            positions: List of position dicts
            
        Returns:
            Total current equity
        """
        unrealized = self.calculate_unrealized_pnl(positions)
        position_value = sum(
            pos.get('current_price', 0) * pos.get('shares', 0)
            for pos in positions
        )
        return cash_balance + position_value
    
    def get_daily_pnl(self, current_equity: float) -> dict:
        """
        Get daily P&L summary.
        
        Args:
            current_equity: Current portfolio equity
            
        Returns:
            Dict with P&L metrics
        """
        if self.starting_equity <= 0:
            return {
                'pnl_usd': 0.0,
                'pnl_pct': 0.0,
                'starting_equity': 0.0,
                'current_equity': current_equity
            }
        
        pnl_usd = current_equity - self.starting_equity
        pnl_pct = pnl_usd / self.starting_equity
        
        return {
            'pnl_usd': pnl_usd,
            'pnl_pct': pnl_pct,
            'starting_equity': self.starting_equity,
            'current_equity': current_equity
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_risk_manager_from_config(config: dict, logger=None) -> RiskManager:
    """
    Create RiskManager from config dictionary.
    
    Args:
        config: Config dict (from config.yaml RISK section)
        logger: Optional logger function
        
    Returns:
        Configured RiskManager instance
    """
    risk_cfg = config.get('RISK', {})
    
    return RiskManager(
        stop_loss_pct=risk_cfg.get('STOP_LOSS_PCT', 0.02),
        max_positions=risk_cfg.get('MAX_POSITIONS', 3),
        daily_loss_limit_pct=risk_cfg.get('DAILY_LOSS_LIMIT_PCT', 0.01),
        target_risk_per_trade_usd=risk_cfg.get('TARGET_RISK_PER_TRADE_USD', 125.0),
        max_position_size_usd=risk_cfg.get('MAX_POSITION_SIZE_USD', 5000.0),
        atr_risk_multiplier=risk_cfg.get('ATR_RISK_MULTIPLIER', 1.5),
        logger=logger
    )
