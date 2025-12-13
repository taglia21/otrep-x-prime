"""
Risk Management Module
======================
Centralized risk management for OTREP-X PRIME.

Components:
- KillSwitch: Global circuit breaker
- PositionSizer: ATR-based volatility targeting
- RiskManager: Position-level risk controls (parent module)
"""

from .kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    KillSwitchStatus,
    KillSwitchReason,
    create_kill_switch_from_config
)

from .position_sizer import (
    PositionSizer,
    PositionSizerConfig,
    PositionSizeResult,
    compute_atr,
    compute_position_size,
)

__all__ = [
    # Kill Switch
    'KillSwitch',
    'KillSwitchConfig', 
    'KillSwitchStatus',
    'KillSwitchReason',
    'create_kill_switch_from_config',
    # Position Sizer
    'PositionSizer',
    'PositionSizerConfig',
    'PositionSizeResult',
    'compute_atr',
    'compute_position_size',
]
