"""
Position Sizer - ATR-Based Volatility Targeting
=================================================
Calculate position sizes based on volatility (ATR) targeting.

CTO Directive (Dec 2025):
- Replace fixed dollar sizing with ATR-based volatility targeting
- Daily budget: $1,000 / 8 expected stop-outs = $125 max risk per trade

Formula:
  risk_per_share = ATR * atr_multiplier
  shares = floor(target_risk_usd / risk_per_share)
  
Author: OTREP-X Development Team
Date: December 2025
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict
import numpy as np


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    shares: int
    notional_usd: float
    expected_risk_usd: float
    atr: float
    risk_per_share: float
    was_capped: bool = False
    cap_reason: Optional[str] = None


@dataclass
class PositionSizerConfig:
    """Configuration for position sizing."""
    # Target risk per trade (CTO: $1000/8 = $125)
    target_risk_per_trade_usd: float = 125.0
    
    # Maximum position size cap
    max_position_size_usd: float = 5000.0
    
    # ATR multiplier (how many ATRs define "risk")
    atr_multiplier: float = 1.5
    
    # ATR period for calculation
    atr_period: int = 14
    
    # Minimum position size (floor)
    min_shares: int = 1
    
    # Daily loss limit used for risk budgeting
    daily_loss_limit_usd: float = 1000.0
    
    # Expected number of stop-outs per day (for risk budgeting)
    expected_stop_outs: int = 8


class PositionSizer:
    """
    ATR-Based Position Sizer.
    
    Uses Average True Range (ATR) to calculate position sizes
    that target a consistent dollar risk per trade.
    
    Example:
        sizer = PositionSizer(config)
        
        # Calculate ATR from recent bars
        atr = sizer.calculate_atr(highs, lows, closes)
        
        # Get position size
        result = sizer.calculate_size(
            entry_price=100.0,
            atr=atr,
            symbol='AAPL'
        )
        
        print(f"Buy {result.shares} shares, risk ${result.expected_risk_usd:.2f}")
    """
    
    def __init__(
        self,
        config: Optional[PositionSizerConfig] = None,
        logger: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize position sizer.
        
        Args:
            config: Position sizing configuration
            logger: Optional logging function
        """
        self.config = config or PositionSizerConfig()
        self.log = logger or (lambda x: None)  # Silent by default
    
    def calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: Optional[int] = None
    ) -> float:
        """
        Calculate Average True Range (ATR).
        
        True Range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        
        ATR = SMA(True Range, period)
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices
            period: ATR period (default: config.atr_period)
            
        Returns:
            ATR value (dollars per share)
        """
        period = period or self.config.atr_period
        
        if len(highs) < period + 1:
            # Not enough data - use simple high-low range
            if len(highs) > 0:
                return float(np.mean(highs - lows))
            return 0.0
        
        # Calculate True Range
        tr = np.zeros(len(highs))
        
        # First bar: just high - low
        tr[0] = highs[0] - lows[0]
        
        # Subsequent bars: max of three ranges
        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hpc = abs(highs[i] - closes[i - 1])
            lpc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hpc, lpc)
        
        # ATR is SMA of True Range over period
        atr = np.mean(tr[-period:])
        
        return float(atr)
    
    def calculate_atr_from_bars(
        self,
        bars: List[Dict],
        period: Optional[int] = None
    ) -> float:
        """
        Calculate ATR from bar dictionaries.
        
        Args:
            bars: List of bar dicts with 'high', 'low', 'close' keys
            period: ATR period
            
        Returns:
            ATR value
        """
        if not bars:
            return 0.0
        
        highs = np.array([b['high'] for b in bars])
        lows = np.array([b['low'] for b in bars])
        closes = np.array([b['close'] for b in bars])
        
        return self.calculate_atr(highs, lows, closes, period)
    
    def calculate_size(
        self,
        entry_price: float,
        atr: float,
        symbol: Optional[str] = None,
        override_risk_usd: Optional[float] = None,
        override_max_usd: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate position size using volatility targeting.
        
        Args:
            entry_price: Expected entry price
            atr: Average True Range for the symbol
            symbol: Symbol name (for logging)
            override_risk_usd: Override target risk per trade
            override_max_usd: Override max position size
            
        Returns:
            PositionSizeResult with shares and risk metrics
        """
        target_risk = override_risk_usd or self.config.target_risk_per_trade_usd
        max_position = override_max_usd or self.config.max_position_size_usd
        
        # Handle edge cases
        if entry_price <= 0:
            self.log(f"⚠️ Invalid entry_price ({entry_price}), using minimum size")
            return PositionSizeResult(
                shares=self.config.min_shares,
                notional_usd=0.0,
                expected_risk_usd=0.0,
                atr=0.0,
                risk_per_share=0.0,
                was_capped=True,
                cap_reason="invalid_entry_price"
            )
        
        if atr <= 0:
            # No ATR available - use 2% of price as proxy
            atr = entry_price * 0.02
            self.log(f"⚠️ ATR not available, using 2% proxy: ${atr:.2f}")
        
        # Calculate dollar risk per share
        risk_per_share = atr * self.config.atr_multiplier
        
        # Calculate raw position size
        raw_shares = target_risk / risk_per_share
        
        # Apply maximum position size cap
        max_shares_by_notional = max_position / entry_price
        
        was_capped = False
        cap_reason = None
        
        if raw_shares > max_shares_by_notional:
            raw_shares = max_shares_by_notional
            was_capped = True
            cap_reason = "max_position_size"
        
        # Floor to integer, minimum 1 share
        final_shares = max(self.config.min_shares, int(math.floor(raw_shares)))
        
        # Calculate actuals
        notional = final_shares * entry_price
        expected_risk = final_shares * risk_per_share
        
        # Log the calculation
        symbol_str = symbol or "UNKNOWN"
        self.log(f"📐 {symbol_str}: {final_shares} shares @ ${entry_price:.2f} = ${notional:.2f}")
        self.log(f"   ATR: ${atr:.2f}, Risk/share: ${risk_per_share:.2f}, Total risk: ${expected_risk:.2f}")
        
        if was_capped:
            self.log(f"   ⚠️ Capped by {cap_reason}")
        
        return PositionSizeResult(
            shares=final_shares,
            notional_usd=notional,
            expected_risk_usd=expected_risk,
            atr=atr,
            risk_per_share=risk_per_share,
            was_capped=was_capped,
            cap_reason=cap_reason
        )
    
    def calculate_risk_budget(
        self,
        daily_limit_usd: Optional[float] = None,
        expected_trades: Optional[int] = None
    ) -> float:
        """
        Calculate per-trade risk budget from daily limit.
        
        CTO Formula:
          risk_per_trade = max(daily_limit / expected_stop_outs, $25 floor)
        
        Args:
            daily_limit_usd: Daily loss budget (default: config value)
            expected_trades: Expected number of stop-outs
            
        Returns:
            Recommended risk per trade in USD
        """
        daily_limit = daily_limit_usd or self.config.daily_loss_limit_usd
        expected = expected_trades or self.config.expected_stop_outs
        
        # Calculate with floor
        risk_per_trade = max(daily_limit / expected, 25.0)
        
        return risk_per_trade
    
    def batch_calculate(
        self,
        positions: List[Dict],
        atr_data: Dict[str, float]
    ) -> Dict[str, PositionSizeResult]:
        """
        Calculate sizes for multiple positions.
        
        Args:
            positions: List of dicts with 'symbol' and 'entry_price'
            atr_data: Dict mapping symbol -> ATR value
            
        Returns:
            Dict mapping symbol -> PositionSizeResult
        """
        results = {}
        
        for pos in positions:
            symbol = pos['symbol']
            entry_price = pos['entry_price']
            atr = atr_data.get(symbol, 0.0)
            
            results[symbol] = self.calculate_size(
                entry_price=entry_price,
                atr=atr,
                symbol=symbol
            )
        
        return results


# =============================================================================
# STANDALONE ATR FUNCTIONS
# =============================================================================

def compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> float:
    """
    Standalone ATR calculation function.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        period: ATR period
        
    Returns:
        ATR value
    """
    sizer = PositionSizer()
    return sizer.calculate_atr(highs, lows, closes, period)


def compute_position_size(
    entry_price: float,
    atr: float,
    target_risk_usd: float = 125.0,
    max_position_usd: float = 5000.0,
    atr_multiplier: float = 1.5
) -> int:
    """
    Standalone position size calculation.
    
    Args:
        entry_price: Expected entry price
        atr: Average True Range
        target_risk_usd: Target dollar risk per trade
        max_position_usd: Maximum position size
        atr_multiplier: ATR multiplier
        
    Returns:
        Number of shares
    """
    config = PositionSizerConfig(
        target_risk_per_trade_usd=target_risk_usd,
        max_position_size_usd=max_position_usd,
        atr_multiplier=atr_multiplier
    )
    sizer = PositionSizer(config)
    result = sizer.calculate_size(entry_price, atr)
    return result.shares
