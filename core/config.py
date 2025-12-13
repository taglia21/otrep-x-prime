"""
Unified Configuration Loader
=============================
Single source of truth for all OTREP-X PRIME configuration.

Features:
- Load from config.yaml
- Environment variable overrides
- Schema validation
- Typed dataclass access
- Backward compatibility with old key names

Author: OTREP-X Development Team
Date: December 2025
"""

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import yaml


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class StrategyConfig:
    """Strategy parameters configuration."""
    timeframe: str = "5Min"
    momentum_lookback: int = 20
    trend_lookback: int = 15
    signal_threshold: float = 0.15
    neutral_threshold: float = 0.05
    
    # Signal weights
    momentum_weight: float = 0.4
    trend_weight: float = 0.0
    mean_reversion_weight: float = 0.4
    graph_weight: float = 0.2
    
    # Adaptive settings
    adaptive_enabled: bool = True
    high_vol_lookback: int = 10
    low_vol_lookback: int = 30
    vol_multiplier: float = 1.5
    
    # Mean reversion settings
    mean_reversion_enabled: bool = True
    mean_reversion_lookback: int = 20
    bb_std_dev_multiplier: float = 2.0
    reversion_threshold: float = 0.01


@dataclass
class GraphAlphaConfig:
    """Graph Alpha (Laplacian diffusion) configuration."""
    enabled: bool = True
    
    # CTO Directive: Use 0.5 threshold for stable adjacency
    # Key migration: CORRELATION_THRESHOLD -> ADJACENCY_THRESHOLD
    adjacency_threshold: float = 0.5
    
    diffusion_alpha: float = 1.0
    lookback_bars: int = 50
    
    # Backward compatibility alias
    @property
    def correlation_threshold(self) -> float:
        """Deprecated: Use adjacency_threshold instead."""
        return self.adjacency_threshold


@dataclass
class MarketFilterConfig:
    """Market regime filter configuration."""
    # SPY correlation filter (separate from graph adjacency)
    spy_correlation_threshold: float = 0.75
    market_proxy_symbol: str = "SPY"
    lookback_days: int = 30


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Volatility targeting (CTO Directive)
    target_risk_per_trade_usd: float = 125.0  # $1000/8 expected stop-outs
    max_position_size_usd: float = 5000.0
    atr_risk_multiplier: float = 1.5
    atr_period: int = 14
    
    # Legacy (deprecated)
    position_size_usd: float = 500.0
    
    # Stop-loss and limits
    stop_loss_pct: float = 0.02
    max_positions: int = 3
    daily_loss_limit_pct: float = 0.01
    
    # Kill switch thresholds
    kill_switch_drawdown_pct: float = 0.004  # 0.40%
    kill_switch_max_rejects: int = 5
    kill_switch_data_stale_sec: float = 600.0  # 10 minutes
    kill_switch_max_api_fails: int = 3


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    
    # Loaded from environment
    api_key: Optional[str] = None
    api_secret: Optional[str] = None


@dataclass
class PolygonConfig:
    """Polygon API configuration."""
    base_url: str = "https://api.polygon.io"
    
    # Loaded from environment
    api_key: Optional[str] = None


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    slippage_bps: float = 5.0
    commission_fixed: float = 1.0
    initial_capital: float = 100000.0


@dataclass
class SystemConfig:
    """System-level configuration."""
    symbols: List[str] = field(default_factory=list)
    check_interval: int = 60


@dataclass
class Config:
    """
    Master configuration container.
    
    Single source of truth for all OTREP-X PRIME settings.
    Loaded from config.yaml with environment variable overrides.
    
    Usage:
        from core.config import load_config
        
        config = load_config('config.yaml')
        
        # Access typed settings
        print(config.strategy.signal_threshold)
        print(config.risk.target_risk_per_trade_usd)
        print(config.graph_alpha.adjacency_threshold)
    """
    system: SystemConfig = field(default_factory=SystemConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    graph_alpha: GraphAlphaConfig = field(default_factory=GraphAlphaConfig)
    market_filter: MarketFilterConfig = field(default_factory=MarketFilterConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    polygon: PolygonConfig = field(default_factory=PolygonConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Convenience aliases for backward compatibility
    @property
    def symbols(self) -> List[str]:
        return self.system.symbols
    
    @property
    def check_interval(self) -> int:
        return self.system.check_interval
    
    @property
    def bar_size(self) -> str:
        return self.strategy.timeframe
    
    @property
    def alpaca_base_url(self) -> str:
        return self.alpaca.base_url
    
    # Graph params for C++ engine
    @property
    def graph_params(self) -> GraphAlphaConfig:
        return self.graph_alpha
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file with env overrides."""
        return load_config(path)


# =============================================================================
# LOADER FUNCTIONS
# =============================================================================

def _get_env_override(key: str, default: Any = None) -> Optional[str]:
    """Get environment variable override for a config key."""
    # Convert YAML key path to env var name
    # e.g., 'RISK.TARGET_RISK_PER_TRADE_USD' -> 'OTREP_RISK_TARGET_RISK_PER_TRADE_USD'
    env_key = f"OTREP_{key.upper().replace('.', '_')}"
    return os.environ.get(env_key, default)


def _migrate_deprecated_keys(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate deprecated config keys with warnings.
    
    Migrations:
    - GRAPH_ALPHA.CORRELATION_THRESHOLD -> GRAPH_ALPHA.ADJACENCY_THRESHOLD
    - RISK.POSITION_SIZE_USD is deprecated (use TARGET_RISK_PER_TRADE_USD)
    """
    graph_alpha = raw.get('GRAPH_ALPHA', {})
    
    # Migrate CORRELATION_THRESHOLD to ADJACENCY_THRESHOLD
    if 'CORRELATION_THRESHOLD' in graph_alpha and 'ADJACENCY_THRESHOLD' not in graph_alpha:
        graph_alpha['ADJACENCY_THRESHOLD'] = graph_alpha['CORRELATION_THRESHOLD']
        warnings.warn(
            "GRAPH_ALPHA.CORRELATION_THRESHOLD is deprecated. "
            "Use GRAPH_ALPHA.ADJACENCY_THRESHOLD instead.",
            DeprecationWarning,
            stacklevel=3
        )
    
    # Warn about deprecated POSITION_SIZE_USD
    risk = raw.get('RISK', {})
    if 'POSITION_SIZE_USD' in risk:
        warnings.warn(
            "RISK.POSITION_SIZE_USD is deprecated. "
            "Use RISK.TARGET_RISK_PER_TRADE_USD with ATR-based sizing instead.",
            DeprecationWarning,
            stacklevel=3
        )
    
    # Migrate MARKET_FILTER.CORRELATION_THRESHOLD to SPY_CORRELATION_THRESHOLD
    market_filter = raw.get('MARKET_FILTER', {})
    if 'CORRELATION_THRESHOLD' in market_filter and 'SPY_CORRELATION_THRESHOLD' not in market_filter:
        market_filter['SPY_CORRELATION_THRESHOLD'] = market_filter['CORRELATION_THRESHOLD']
        # Don't warn - this is more of a rename for clarity
    
    raw['GRAPH_ALPHA'] = graph_alpha
    raw['RISK'] = risk
    raw['MARKET_FILTER'] = market_filter
    
    return raw


def _parse_strategy(raw: Dict[str, Any]) -> StrategyConfig:
    """Parse STRATEGY section into typed config."""
    strategy = raw.get('STRATEGY', {})
    weights = strategy.get('SIGNAL_WEIGHTS', {})
    adaptive = strategy.get('ADAPTIVE', {})
    mr = strategy.get('MEAN_REVERSION', {})
    
    return StrategyConfig(
        timeframe=strategy.get('TIMEFRAME', '5Min'),
        momentum_lookback=strategy.get('MOMENTUM_LOOKBACK', 20),
        trend_lookback=strategy.get('TREND_LOOKBACK', 15),
        signal_threshold=strategy.get('SIGNAL_THRESHOLD', 0.15),
        neutral_threshold=strategy.get('NEUTRAL_THRESHOLD', 0.05),
        momentum_weight=weights.get('MOMENTUM', 0.4),
        trend_weight=weights.get('TREND', 0.0),
        mean_reversion_weight=weights.get('MEAN_REVERSION', 0.4),
        graph_weight=weights.get('GRAPH', 0.2),
        adaptive_enabled=adaptive.get('ENABLED', True),
        high_vol_lookback=adaptive.get('HIGH_VOL_LOOKBACK', 10),
        low_vol_lookback=adaptive.get('LOW_VOL_LOOKBACK', 30),
        vol_multiplier=adaptive.get('VOL_MULTIPLIER', 1.5),
        mean_reversion_enabled=mr.get('ENABLED', True),
        mean_reversion_lookback=mr.get('LOOKBACK', 20),
        bb_std_dev_multiplier=mr.get('BB_STD_DEV_MULTIPLIER', 2.0),
        reversion_threshold=mr.get('REVERSION_THRESHOLD', 0.01),
    )


def _parse_graph_alpha(raw: Dict[str, Any]) -> GraphAlphaConfig:
    """Parse GRAPH_ALPHA section into typed config."""
    ga = raw.get('GRAPH_ALPHA', {})
    
    # Support both old and new key names
    threshold = ga.get('ADJACENCY_THRESHOLD', ga.get('CORRELATION_THRESHOLD', 0.5))
    
    return GraphAlphaConfig(
        enabled=ga.get('ENABLED', True),
        adjacency_threshold=threshold,
        diffusion_alpha=ga.get('DIFFUSION_ALPHA', 1.0),
        lookback_bars=ga.get('LOOKBACK_BARS', 50),
    )


def _parse_market_filter(raw: Dict[str, Any]) -> MarketFilterConfig:
    """Parse MARKET_FILTER section into typed config."""
    mf = raw.get('MARKET_FILTER', {})
    
    # Support both old and new key names
    threshold = mf.get('SPY_CORRELATION_THRESHOLD', mf.get('CORRELATION_THRESHOLD', 0.75))
    
    return MarketFilterConfig(
        spy_correlation_threshold=threshold,
        market_proxy_symbol=mf.get('MARKET_PROXY_SYMBOL', 'SPY'),
        lookback_days=mf.get('LOOKBACK_DAYS', 30),
    )


def _parse_risk(raw: Dict[str, Any]) -> RiskConfig:
    """Parse RISK section into typed config."""
    risk = raw.get('RISK', {})
    
    return RiskConfig(
        target_risk_per_trade_usd=risk.get('TARGET_RISK_PER_TRADE_USD', 125.0),
        max_position_size_usd=risk.get('MAX_POSITION_SIZE_USD', 5000.0),
        atr_risk_multiplier=risk.get('ATR_RISK_MULTIPLIER', 1.5),
        atr_period=risk.get('ATR_PERIOD', 14),
        position_size_usd=risk.get('POSITION_SIZE_USD', 500.0),
        stop_loss_pct=risk.get('STOP_LOSS_PCT', 0.02),
        max_positions=risk.get('MAX_POSITIONS', 3),
        daily_loss_limit_pct=risk.get('DAILY_LOSS_LIMIT_PCT', 0.01),
        kill_switch_drawdown_pct=risk.get('KILL_SWITCH_DRAWDOWN_PCT', 0.004),
        kill_switch_max_rejects=risk.get('KILL_SWITCH_MAX_REJECTS', 5),
        kill_switch_data_stale_sec=risk.get('KILL_SWITCH_DATA_STALE_SEC', 600.0),
        kill_switch_max_api_fails=risk.get('KILL_SWITCH_MAX_API_FAILS', 3),
    )


def _parse_system(raw: Dict[str, Any]) -> SystemConfig:
    """Parse SYSTEM section into typed config."""
    system = raw.get('SYSTEM', {})
    
    return SystemConfig(
        symbols=system.get('SYMBOLS', []),
        check_interval=system.get('CHECK_INTERVAL', 60),
    )


def _parse_alpaca(raw: Dict[str, Any]) -> AlpacaConfig:
    """Parse ALPACA section with env overrides."""
    alpaca = raw.get('ALPACA', {})

    api_key = (
        os.environ.get('APCA_API_KEY_ID')
        or os.environ.get('ALPACA_API_KEY_ID')
        or os.environ.get('ALPACA_API_KEY')
    )
    api_secret = (
        os.environ.get('APCA_API_SECRET_KEY')
        or os.environ.get('ALPACA_API_SECRET_KEY')
        or os.environ.get('ALPACA_SECRET_KEY')
        or os.environ.get('ALPACA_API_SECRET')
    )
    
    return AlpacaConfig(
        base_url=alpaca.get('BASE_URL', 'https://paper-api.alpaca.markets'),
        data_url=alpaca.get('DATA_URL', 'https://data.alpaca.markets'),
        api_key=api_key,
        api_secret=api_secret,
    )


def _parse_polygon(raw: Dict[str, Any]) -> PolygonConfig:
    """Parse POLYGON section with env overrides."""
    polygon = raw.get('POLYGON', {})
    
    return PolygonConfig(
        base_url=polygon.get('BASE_URL', 'https://api.polygon.io'),
        api_key=os.environ.get('POLYGON_API_KEY'),
    )


def _parse_backtest(raw: Dict[str, Any]) -> BacktestConfig:
    """Parse BACKTEST section into typed config."""
    bt = raw.get('BACKTEST', {})
    
    return BacktestConfig(
        slippage_bps=bt.get('SLIPPAGE_BPS', 5.0),
        commission_fixed=bt.get('COMMISSION_FIXED', 1.0),
        initial_capital=bt.get('INITIAL_CAPITAL', 100000.0),
    )


def _validate_config(config: Config) -> None:
    """
    Validate configuration for obvious errors.
    
    Raises:
        ValueError: If configuration is invalid
    """
    errors = []
    
    # Check symbols
    if not config.system.symbols:
        errors.append("SYSTEM.SYMBOLS cannot be empty")
    
    # Check signal weights sum
    total_weight = (
        config.strategy.momentum_weight +
        config.strategy.trend_weight +
        config.strategy.mean_reversion_weight +
        config.strategy.graph_weight
    )
    if abs(total_weight - 1.0) > 0.01:
        warnings.warn(
            f"Signal weights sum to {total_weight:.2f}, not 1.0. "
            "This may lead to unexpected signal magnitudes.",
            UserWarning
        )
    
    # Check risk params
    if config.risk.target_risk_per_trade_usd <= 0:
        errors.append("RISK.TARGET_RISK_PER_TRADE_USD must be positive")
    
    if config.risk.max_position_size_usd <= 0:
        errors.append("RISK.MAX_POSITION_SIZE_USD must be positive")
    
    # Check graph params
    if not 0.0 < config.graph_alpha.adjacency_threshold < 1.0:
        errors.append("GRAPH_ALPHA.ADJACENCY_THRESHOLD must be between 0 and 1")
    
    # Check kill switch params
    if config.risk.kill_switch_drawdown_pct <= 0:
        errors.append("RISK.KILL_SWITCH_DRAWDOWN_PCT must be positive")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))


def load_config(path: Union[str, Path] = 'config.yaml') -> Config:
    """
    Load and parse configuration from YAML file.
    
    Args:
        path: Path to config.yaml file
        
    Returns:
        Validated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    
    # Migrate deprecated keys
    raw = _migrate_deprecated_keys(raw)
    
    # Parse all sections
    config = Config(
        system=_parse_system(raw),
        strategy=_parse_strategy(raw),
        graph_alpha=_parse_graph_alpha(raw),
        market_filter=_parse_market_filter(raw),
        risk=_parse_risk(raw),
        alpaca=_parse_alpaca(raw),
        polygon=_parse_polygon(raw),
        backtest=_parse_backtest(raw),
    )
    
    # Validate
    _validate_config(config)
    
    return config


# =============================================================================
# CONVENIENCE
# =============================================================================

def get_default_config() -> Config:
    """Get config with all defaults (no YAML file needed)."""
    return Config(
        system=SystemConfig(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
            check_interval=60
        )
    )
