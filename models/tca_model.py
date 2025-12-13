"""
Transaction Cost Analysis (TCA) Model
=====================================
Realistic cost modeling for backtesting and execution analysis.

Cost Components:
1. Slippage: Market impact from order execution
2. Commission: Fixed + variable broker fees
3. Spread: Bid-ask spread cost

Author: OTREP-X Development Team
Date: December 2025
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OrderType(Enum):
    """Order execution type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class LiquidityProfile(Enum):
    """Asset liquidity profile."""
    HIGH = "high"      # AAPL, MSFT, etc.
    MEDIUM = "medium"  # Mid-cap stocks
    LOW = "low"        # Small-cap, illiquid


@dataclass
class TCAConfig:
    """TCA model configuration."""
    # Slippage in basis points (default: 5 bps for liquid stocks)
    # CTO Reference: Typical HFT slippage is 2-10 bps
    slippage_bps_high_liquidity: float = 3.0
    slippage_bps_medium_liquidity: float = 8.0
    slippage_bps_low_liquidity: float = 20.0
    
    # Commission structure (Alpaca: $0 for stocks)
    # But model realistic costs for proper backtesting
    commission_fixed_usd: float = 0.0    # Fixed per trade
    commission_per_share: float = 0.0    # Per share (some brokers)
    commission_pct: float = 0.0          # Percentage of notional
    
    # Bid-ask spread in basis points
    spread_bps_high_liquidity: float = 1.0
    spread_bps_medium_liquidity: float = 5.0
    spread_bps_low_liquidity: float = 15.0
    
    # Market impact (additional slippage for large orders)
    # Impact = impact_coefficient * sqrt(size_pct)
    impact_coefficient: float = 0.1
    
    # SEC fee (for sell orders only)
    sec_fee_rate: float = 0.0000278  # $27.80 per $1,000,000


@dataclass
class TCAResult:
    """Transaction cost breakdown."""
    gross_price: float           # Execution price before costs
    net_price: float             # Effective price after costs
    total_cost_usd: float        # Total cost in dollars
    total_cost_bps: float        # Total cost in basis points
    
    slippage_usd: float
    spread_usd: float
    commission_usd: float
    market_impact_usd: float
    sec_fee_usd: float
    
    def __repr__(self) -> str:
        return (f"TCAResult(net_price=${self.net_price:.4f}, "
                f"total_cost=${self.total_cost_usd:.4f} ({self.total_cost_bps:.2f}bps))")


class TCAModel:
    """
    Transaction Cost Analysis Model.
    
    Calculates realistic execution costs for backtesting and live trading.
    
    Usage:
        tca = TCAModel()
        result = tca.calculate_costs(
            price=150.00,
            size=100,
            side='buy',
            order_type=OrderType.MARKET
        )
        print(f"Effective price: ${result.net_price:.2f}")
        print(f"Total cost: ${result.total_cost_usd:.2f}")
    """
    
    def __init__(
        self,
        config: Optional[TCAConfig] = None,
        liquidity_profile: LiquidityProfile = LiquidityProfile.HIGH
    ):
        """
        Initialize TCA model.
        
        Args:
            config: TCA configuration (uses defaults if None)
            liquidity_profile: Default liquidity profile for assets
        """
        self.config = config or TCAConfig()
        self.default_liquidity = liquidity_profile
    
    def calculate_costs(
        self,
        price: float,
        size: int,
        side: str,
        order_type: OrderType = OrderType.MARKET,
        liquidity: Optional[LiquidityProfile] = None,
        adv: Optional[float] = None  # Average daily volume
    ) -> TCAResult:
        """
        Calculate transaction costs for an order.
        
        Args:
            price: Execution price
            size: Number of shares
            side: 'buy' or 'sell'
            order_type: Order type (MARKET, LIMIT, STOP)
            liquidity: Liquidity profile (overrides default)
            adv: Average daily volume for market impact calculation
            
        Returns:
            TCAResult with cost breakdown
        """
        liq = liquidity or self.default_liquidity
        notional = price * size
        is_buy = side.lower() == 'buy'
        
        # 1. Slippage (market orders have higher slippage)
        slippage_bps = self._get_slippage_bps(liq, order_type)
        slippage_usd = notional * (slippage_bps / 10000)
        
        # 2. Bid-ask spread (always pay half the spread)
        spread_bps = self._get_spread_bps(liq)
        spread_usd = notional * (spread_bps / 10000 / 2)
        
        # 3. Commission
        commission_usd = (
            self.config.commission_fixed_usd +
            self.config.commission_per_share * size +
            notional * self.config.commission_pct
        )
        
        # 4. Market impact (for large orders)
        market_impact_usd = 0.0
        if adv and adv > 0:
            size_pct = size / adv
            if size_pct > 0.001:  # Only applies if > 0.1% of ADV
                impact_bps = self.config.impact_coefficient * (size_pct ** 0.5) * 10000
                market_impact_usd = notional * (impact_bps / 10000)
        
        # 5. SEC fee (sell orders only)
        sec_fee_usd = 0.0
        if not is_buy:
            sec_fee_usd = notional * self.config.sec_fee_rate
        
        # Total cost
        total_cost_usd = slippage_usd + spread_usd + commission_usd + market_impact_usd + sec_fee_usd
        total_cost_bps = (total_cost_usd / notional) * 10000 if notional > 0 else 0
        
        # Net price (worse for the trader)
        if is_buy:
            net_price = price + (total_cost_usd / size) if size > 0 else price
        else:
            net_price = price - (total_cost_usd / size) if size > 0 else price
        
        return TCAResult(
            gross_price=price,
            net_price=net_price,
            total_cost_usd=total_cost_usd,
            total_cost_bps=total_cost_bps,
            slippage_usd=slippage_usd,
            spread_usd=spread_usd,
            commission_usd=commission_usd,
            market_impact_usd=market_impact_usd,
            sec_fee_usd=sec_fee_usd
        )
    
    def _get_slippage_bps(self, liquidity: LiquidityProfile, order_type: OrderType) -> float:
        """Get slippage in basis points based on liquidity and order type."""
        base_slippage = {
            LiquidityProfile.HIGH: self.config.slippage_bps_high_liquidity,
            LiquidityProfile.MEDIUM: self.config.slippage_bps_medium_liquidity,
            LiquidityProfile.LOW: self.config.slippage_bps_low_liquidity
        }[liquidity]
        
        # Limit orders have lower slippage (assuming good execution)
        if order_type == OrderType.LIMIT:
            return base_slippage * 0.3
        elif order_type == OrderType.STOP:
            return base_slippage * 1.5  # Stop orders often have worse fills
        
        return base_slippage
    
    def _get_spread_bps(self, liquidity: LiquidityProfile) -> float:
        """Get bid-ask spread in basis points."""
        return {
            LiquidityProfile.HIGH: self.config.spread_bps_high_liquidity,
            LiquidityProfile.MEDIUM: self.config.spread_bps_medium_liquidity,
            LiquidityProfile.LOW: self.config.spread_bps_low_liquidity
        }[liquidity]


def calculate_slippage_and_commission(
    price: float,
    size: int,
    order_type: str = 'market',
    slippage_bps: float = 5.0,
    commission_fixed: float = 0.0
) -> dict:
    """
    Simple function to calculate slippage and commission.
    
    This is a convenience function for basic TCA in backtesting.
    
    Args:
        price: Execution price
        size: Number of shares
        order_type: 'market', 'limit', or 'stop'
        slippage_bps: Slippage in basis points (default: 5 bps)
        commission_fixed: Fixed commission per trade (default: $0)
        
    Returns:
        Dict with slippage_usd, commission_usd, total_cost_usd, net_price
    
    Example:
        >>> costs = calculate_slippage_and_commission(150.00, 100)
        >>> print(f"Total cost: ${costs['total_cost_usd']:.2f}")
        Total cost: $7.50
    """
    notional = price * size
    
    # Adjust slippage by order type
    if order_type.lower() == 'limit':
        effective_slippage_bps = slippage_bps * 0.3
    elif order_type.lower() == 'stop':
        effective_slippage_bps = slippage_bps * 1.5
    else:
        effective_slippage_bps = slippage_bps
    
    slippage_usd = notional * (effective_slippage_bps / 10000)
    commission_usd = commission_fixed
    total_cost_usd = slippage_usd + commission_usd
    
    # Assuming buy order (cost increases effective price)
    net_price = price + (total_cost_usd / size) if size > 0 else price
    
    return {
        'slippage_usd': slippage_usd,
        'commission_usd': commission_usd,
        'total_cost_usd': total_cost_usd,
        'net_price': net_price,
        'cost_bps': (total_cost_usd / notional) * 10000 if notional > 0 else 0
    }
