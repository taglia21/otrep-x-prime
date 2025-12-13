"""
OTREP-X PRIME - Minimal Viable Trader (MVT) v5.0 (Graph Alpha Engine)
======================================================================
Modular controller integrating:
- C++ HybridStrategy engine (27x faster than Python)
- C++ MarketGraph engine (Laplacian diffusion signals)
- Alpaca API for execution
- Polygon API for data (with Alpaca fallback)
- Correlation-based market filter
- Essential risk management

Author: OTREP-X Development Team
Lead Engineer: Gemini AI
Date: December 2025
"""

import os
import sys
import time
import yaml
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass

# Local imports
from api.alpaca_client import AlpacaClient
from api.polygon_client import PolygonClient
from risk_manager import RiskManager

# C++ Core import with Python fallback
try:
    import otrep_core
    
    # CRITICAL: Verify the C++ module loaded correctly with all required classes
    if not hasattr(otrep_core, 'HybridStrategy'):
        raise ImportError("otrep_core module loaded, but HybridStrategy class is missing. Did you recompile?")
    
    if not hasattr(otrep_core, 'MarketGraph'):
        raise ImportError("otrep_core module loaded, but MarketGraph class is missing. Did you recompile?")
    
    if not hasattr(otrep_core, 'GraphParams'):
        raise ImportError("otrep_core module loaded, but GraphParams class is missing. Did you recompile?")
    
    # Verify version is 2.0.0+ (Graph Alpha support)
    version = getattr(otrep_core, '__version__', '0.0.0')
    major_version = int(version.split('.')[0])
    if major_version < 2:
        raise ImportError(f"otrep_core version {version} too old. Need v2.0.0+ for Graph Alpha. Recompile required.")
    
    CPP_ENGINE_AVAILABLE = True
    print(f"✅ C++ otrep_core v{version} loaded successfully")
    
except ImportError as e:
    print(f"🔥 FATAL: C++ core failed to load: {e}")
    print("   Run: cd cpp/build && cmake .. && make -j4")
    CPP_ENGINE_AVAILABLE = False
    # Graph Alpha requires C++ - cannot fall back to Python
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    """Typed configuration container."""
    # System
    symbols: List[str]
    check_interval: int
    
    # Alpaca
    alpaca_base_url: str
    alpaca_data_url: str
    
    # Polygon
    polygon_base_url: str
    
    # Strategy
    timeframe: str
    momentum_lookback: int
    trend_lookback: int
    signal_threshold: float
    neutral_threshold: float
    momentum_weight: float
    trend_weight: float
    mean_reversion_weight: float
    adaptive_enabled: bool
    high_vol_lookback: int
    low_vol_lookback: int
    vol_multiplier: float
    # Mean Reversion
    mean_reversion_enabled: bool
    mean_reversion_lookback: int
    bb_std_dev_multiplier: float
    reversion_threshold: float
    
    # Graph Alpha
    graph_alpha_enabled: bool
    graph_weight: float
    graph_correlation_threshold: float
    graph_diffusion_alpha: float
    graph_lookback_bars: int
    
    # Market Filter
    correlation_threshold: float
    market_proxy_symbol: str
    correlation_lookback_days: int
    
    # Risk
    position_size_usd: float
    stop_loss_pct: float
    max_positions: int
    daily_loss_limit_pct: float
    
    @classmethod
    def from_yaml(cls, config_path: str = 'config.yaml') -> 'Config':
        """Create Config instance from YAML file."""
        raw = load_config(config_path)
        
        # Handle optional MARKET_FILTER section
        market_filter = raw.get('MARKET_FILTER', {})
        
        return cls(
            # System
            symbols=raw['SYSTEM']['SYMBOLS'],
            check_interval=raw['SYSTEM']['CHECK_INTERVAL'],
            # Alpaca
            alpaca_base_url=raw['ALPACA']['BASE_URL'],
            alpaca_data_url=raw['ALPACA']['DATA_URL'],
            # Polygon
            polygon_base_url=raw.get('POLYGON', {}).get('BASE_URL', 'https://api.polygon.io'),
            # Strategy
            timeframe=raw['STRATEGY']['TIMEFRAME'],
            momentum_lookback=raw['STRATEGY']['MOMENTUM_LOOKBACK'],
            trend_lookback=raw['STRATEGY']['TREND_LOOKBACK'],
            signal_threshold=raw['STRATEGY']['SIGNAL_THRESHOLD'],
            neutral_threshold=raw['STRATEGY'].get('NEUTRAL_THRESHOLD', 0.05),
            momentum_weight=raw['STRATEGY']['SIGNAL_WEIGHTS']['MOMENTUM'],
            trend_weight=raw['STRATEGY']['SIGNAL_WEIGHTS'].get('TREND', 0.0),
            mean_reversion_weight=raw['STRATEGY']['SIGNAL_WEIGHTS'].get('MEAN_REVERSION', 0.4),
            adaptive_enabled=raw['STRATEGY']['ADAPTIVE']['ENABLED'],
            high_vol_lookback=raw['STRATEGY']['ADAPTIVE']['HIGH_VOL_LOOKBACK'],
            low_vol_lookback=raw['STRATEGY']['ADAPTIVE']['LOW_VOL_LOOKBACK'],
            vol_multiplier=raw['STRATEGY']['ADAPTIVE']['VOL_MULTIPLIER'],
            # Mean Reversion
            mean_reversion_enabled=raw['STRATEGY'].get('MEAN_REVERSION', {}).get('ENABLED', True),
            mean_reversion_lookback=raw['STRATEGY'].get('MEAN_REVERSION', {}).get('LOOKBACK', 20),
            bb_std_dev_multiplier=raw['STRATEGY'].get('MEAN_REVERSION', {}).get('BB_STD_DEV_MULTIPLIER', 2.0),
            reversion_threshold=raw['STRATEGY'].get('MEAN_REVERSION', {}).get('REVERSION_THRESHOLD', 0.01),
            # Graph Alpha
            graph_alpha_enabled=raw.get('GRAPH_ALPHA', {}).get('ENABLED', True),
            graph_weight=raw['STRATEGY']['SIGNAL_WEIGHTS'].get('GRAPH', 0.2),
            graph_correlation_threshold=raw.get('GRAPH_ALPHA', {}).get('CORRELATION_THRESHOLD', 0.5),
            graph_diffusion_alpha=raw.get('GRAPH_ALPHA', {}).get('DIFFUSION_ALPHA', 1.0),
            graph_lookback_bars=raw.get('GRAPH_ALPHA', {}).get('LOOKBACK_BARS', 50),
            # Market Filter
            correlation_threshold=market_filter.get('CORRELATION_THRESHOLD', 0.75),
            market_proxy_symbol=market_filter.get('MARKET_PROXY_SYMBOL', 'SPY'),
            correlation_lookback_days=market_filter.get('LOOKBACK_DAYS', 30),
            # Risk
            position_size_usd=raw['RISK']['POSITION_SIZE_USD'],
            stop_loss_pct=raw['RISK']['STOP_LOSS_PCT'],
            max_positions=raw['RISK']['MAX_POSITIONS'],
            daily_loss_limit_pct=raw['RISK']['DAILY_LOSS_LIMIT_PCT'],
        )


# =============================================================================
# MVT TRADER (MAIN CONTROLLER)
# =============================================================================

class MVTTrader:
    """
    Minimal Viable Trader v5.0 - Graph Alpha Engine
    
    Integrates:
    - C++ HybridStrategy engine (27x faster signal calculations)
    - C++ MarketGraph engine (Laplacian diffusion graph signals)
    - Polygon data client (with Alpaca fallback)
    - Alpaca execution client
    - Correlation-based market filter
    - Risk management
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize API clients
        self.data_client = PolygonClient(
            alpaca_fallback_url=config.alpaca_data_url
        )
        self.exec_client = AlpacaClient(base_url=config.alpaca_base_url)
        
        # Initialize strategy - prefer C++ engine
        if CPP_ENGINE_AVAILABLE:
            # Configure C++ strategy with Phase IV optimized parameters
            params = otrep_core.StrategyParams()
            params.momentum_lookback = config.momentum_lookback
            params.trend_lookback = config.trend_lookback
            params.signal_threshold = config.signal_threshold
            params.neutral_threshold = config.neutral_threshold
            params.momentum_weight = config.momentum_weight
            params.trend_weight = config.trend_weight
            params.adaptive_enabled = config.adaptive_enabled
            params.high_vol_lookback = config.high_vol_lookback
            params.low_vol_lookback = config.low_vol_lookback
            params.vol_multiplier = config.vol_multiplier
            params.mean_reversion_enabled = config.mean_reversion_enabled
            params.mean_reversion_weight = config.mean_reversion_weight
            params.mean_reversion_lookback = config.mean_reversion_lookback
            params.bb_std_dev_multiplier = config.bb_std_dev_multiplier
            params.reversion_threshold = config.reversion_threshold
            params.graph_weight = config.graph_weight
            
            self.strategy = otrep_core.HybridStrategy(params)
            self.use_cpp_engine = True
            self.log("🚀 C++ HybridStrategy engine initialized (27x faster)")
            
            # Initialize MarketGraph engine for graph alpha
            if config.graph_alpha_enabled:
                graph_params = otrep_core.GraphParams()
                graph_params.correlation_threshold = config.graph_correlation_threshold
                graph_params.diffusion_alpha = config.graph_diffusion_alpha
                graph_params.lookback_bars = config.graph_lookback_bars
                self.market_graph = otrep_core.MarketGraph(graph_params)
                self.log("📊 C++ MarketGraph engine initialized (Laplacian diffusion)")
            else:
                self.market_graph = None
        else:
            # Fallback to Python strategy
            self.strategy = PythonHybridStrategy(
                momentum_lookback=config.momentum_lookback,
                trend_lookback=config.trend_lookback,
                signal_threshold=config.signal_threshold,
                neutral_threshold=config.neutral_threshold,
                momentum_weight=config.momentum_weight,
                trend_weight=config.trend_weight,
                adaptive_enabled=config.adaptive_enabled,
                high_vol_lookback=config.high_vol_lookback,
                low_vol_lookback=config.low_vol_lookback,
                vol_multiplier=config.vol_multiplier,
                mean_reversion_enabled=config.mean_reversion_enabled,
                mean_reversion_weight=config.mean_reversion_weight,
                mean_reversion_lookback=config.mean_reversion_lookback,
                bb_std_dev_multiplier=config.bb_std_dev_multiplier,
                reversion_threshold=config.reversion_threshold
            )
            self.use_cpp_engine = False
            self.market_graph = None  # Python fallback doesn't have graph alpha
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            stop_loss_pct=config.stop_loss_pct,
            max_positions=config.max_positions,
            daily_loss_limit_pct=config.daily_loss_limit_pct,
            logger=self.log
        )
        
        # State
        self.active_symbols: List[str] = []
        self.last_signals: Dict[str, float] = {}
        self.cycle_count: int = 0
        self.correlation_cache: Dict[str, float] = {}
        
        # Graph alpha signals (computed once per cycle for all symbols)
        self.graph_signals: Dict[str, float] = {}
        
        # Position tracking (synchronized from broker)
        self.open_positions: Dict[str, Dict] = {}
        
        # Per-symbol C++ strategy instances (since C++ engine is stateful per symbol)
        self.cpp_strategies: Dict[str, 'otrep_core.HybridStrategy'] = {}
        
        # Cache for historical close prices (for graph computation)
        self.price_history: Dict[str, List[float]] = {}
    
    def _get_cpp_strategy(self, symbol: str) -> 'otrep_core.HybridStrategy':
        """Get or create a C++ strategy instance for a symbol."""
        if symbol not in self.cpp_strategies:
            params = otrep_core.StrategyParams()
            params.momentum_lookback = self.config.momentum_lookback
            params.trend_lookback = self.config.trend_lookback
            params.signal_threshold = self.config.signal_threshold
            params.neutral_threshold = self.config.neutral_threshold
            params.momentum_weight = self.config.momentum_weight
            params.trend_weight = self.config.trend_weight
            params.adaptive_enabled = self.config.adaptive_enabled
            params.high_vol_lookback = self.config.high_vol_lookback
            params.low_vol_lookback = self.config.low_vol_lookback
            params.vol_multiplier = self.config.vol_multiplier
            params.mean_reversion_enabled = self.config.mean_reversion_enabled
            params.mean_reversion_weight = self.config.mean_reversion_weight
            params.mean_reversion_lookback = self.config.mean_reversion_lookback
            params.bb_std_dev_multiplier = self.config.bb_std_dev_multiplier
            params.reversion_threshold = self.config.reversion_threshold
            params.graph_weight = self.config.graph_weight
            self.cpp_strategies[symbol] = otrep_core.HybridStrategy(params)
        return self.cpp_strategies[symbol]
    
    def compute_graph_signals(self) -> Dict[str, float]:
        """
        Compute Laplacian diffusion graph signals for all active symbols.
        
        This fetches recent close prices for all symbols, computes log returns,
        and passes them to the C++ MarketGraph engine.
        
        Returns:
            Dict mapping symbol -> graph signal
        """
        if not self.market_graph or not self.use_cpp_engine:
            return {}
        
        if len(self.active_symbols) < 3:
            self.log("📈 Graph alpha: Too few symbols for graph computation")
            return {}
        
        try:
            # Fetch close prices for all active symbols
            closes_dict: Dict[str, List[float]] = {}
            min_bars = self.config.graph_lookback_bars
            
            for symbol in self.active_symbols:
                try:
                    df = self.data_client.get_bars(
                        symbol=symbol,
                        timeframe=self.config.timeframe,
                        limit=min_bars + 5  # Buffer
                    )
                    if len(df) >= min_bars:
                        closes_dict[symbol] = df['close'].tolist()[-min_bars:]
                except Exception:
                    pass
            
            # Need at least 3 symbols for meaningful graph
            valid_symbols = [s for s in self.active_symbols if s in closes_dict]
            if len(valid_symbols) < 3:
                self.log("📈 Graph alpha: Insufficient data for graph")
                return {}
            
            # Build returns matrix (T x N)
            T = min(len(closes_dict[s]) for s in valid_symbols)
            N = len(valid_symbols)
            
            prices = np.zeros((T, N))
            for i, symbol in enumerate(valid_symbols):
                prices[:, i] = closes_dict[symbol][:T]
            
            # Compute log returns
            returns = np.diff(np.log(prices), axis=0)
            
            # Call C++ MarketGraph engine
            result = self.market_graph.calculate_signals(returns)
            
            # Map signals back to symbols
            graph_signals = {}
            for i, symbol in enumerate(valid_symbols):
                if i < len(result.signals):
                    graph_signals[symbol] = float(result.signals[i])
            
            self.log(f"📈 Graph alpha computed: {result.num_edges} edges, "
                    f"avg_corr={result.avg_correlation:.3f}, "
                    f"latency={result.latency_ns/1e6:.2f}ms")
            
            return graph_signals
            
        except Exception as e:
            self.log(f"ERROR computing graph signals: {e}")
            return {}
    
    def sync_positions(self) -> int:
        """
        Synchronize open positions from broker.
        
        This method fetches the TRUE state of all open positions from Alpaca
        and updates the local tracking dictionary. This ensures position count
        is accurate before any trade decisions are made.
        
        Returns:
            Current count of open positions
        """
        try:
            positions = self.exec_client.get_positions()
            
            # Clear and rebuild position tracking
            self.open_positions.clear()
            
            for pos in positions:
                symbol = pos['symbol']
                self.open_positions[symbol] = {
                    'qty': int(pos['qty']),
                    'entry_price': float(pos['avg_entry_price']),
                    'current_price': float(pos['current_price']),
                    'market_value': float(pos['market_value']),
                    'unrealized_pl': float(pos['unrealized_pl']),
                    'unrealized_pl_pct': float(pos['unrealized_plpc'])
                }
            
            position_count = len(self.open_positions)
            self.log(f"📊 Synced positions: {position_count}/{self.config.max_positions}")
            
            if position_count > 0:
                symbols = ', '.join(self.open_positions.keys())
                self.log(f"   Open: {symbols}")
            
            return position_count
            
        except Exception as e:
            self.log(f"ERROR syncing positions: {e}")
            return len(self.open_positions)  # Return cached count on error
    
    def log(self, message: str) -> None:
        """Log message to console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        
        try:
            os.makedirs('logs', exist_ok=True)
            with open('logs/mvt.log', 'a') as f:
                f.write(log_line + '\n')
        except Exception:
            pass
    
    def apply_correlation_filter(self) -> List[str]:
        """
        Filter symbols by correlation to market proxy.
        
        Only trade symbols that have sufficient correlation
        to the broader market (market topology proxy).
        
        Returns:
            List of symbols passing correlation filter
        """
        self.log("\n📊 Applying correlation filter...")
        
        filtered = []
        
        for symbol in self.config.symbols:
            # Check cache first
            if symbol in self.correlation_cache:
                corr = self.correlation_cache[symbol]
            else:
                corr = self.data_client.calculate_correlation(
                    symbol=symbol,
                    market_symbol=self.config.market_proxy_symbol,
                    lookback_days=self.config.correlation_lookback_days
                )
                self.correlation_cache[symbol] = corr
            
            if abs(corr) >= self.config.correlation_threshold:
                filtered.append(symbol)
                self.log(f"  ✓ {symbol}: ρ = {corr:.3f}")
            else:
                self.log(f"  ✗ {symbol}: ρ = {corr:.3f} (below {self.config.correlation_threshold})")
        
        self.log(f"📊 Filtered: {len(filtered)}/{len(self.config.symbols)} symbols\n")
        
        return filtered
    
    def process_symbol(self, symbol: str) -> None:
        """
        Transactional symbol processing with synchronized position tracking.
        
        Uses C++ engine for high-performance signal calculation and
        synchronized position state for accurate risk management.
        """
        try:
            # === STEP 1: Determine Current Position Status ===
            in_position = symbol in self.open_positions
            position_data = self.open_positions.get(symbol)
            current_qty = position_data['qty'] if position_data else 0
            entry_price = position_data['entry_price'] if position_data else 0.0
            
            # === STEP 2: Fetch Market Data ===
            df = self.data_client.get_bars(
                symbol=symbol,
                timeframe=self.config.timeframe,
                limit=50
            )
            
            if len(df) < self.config.momentum_lookback:
                self.log(f"{symbol}: Not enough data ({len(df)} bars)")
                return
            
            current_price = float(df['close'].iloc[-1])
            
            # === STEP 3: Get Graph Signal for this symbol ===
            graph_signal = self.graph_signals.get(symbol, 0.0)
            
            # === STEP 4: Get C++ Signal (HOT PATH) ===
            if self.use_cpp_engine:
                cpp_strategy = self._get_cpp_strategy(symbol)
                cpp_strategy.clear()
                
                for i in range(len(df)):
                    bar = otrep_core.Bar(
                        float(df['open'].iloc[i]),
                        float(df['high'].iloc[i]),
                        float(df['low'].iloc[i]),
                        float(df['close'].iloc[i]),
                        int(df['volume'].iloc[i])
                    )
                    cpp_strategy.on_bar(bar)
                
                # Pass graph signal to strategy
                result = cpp_strategy.calculate_signal(graph_signal)
                signal = result.signal
                lookback_used = result.lookback_used
                volatility = result.volatility
                latency_us = result.latency_ns / 1000.0
                
                action = cpp_strategy.get_trade_action(signal)
                if action == otrep_core.TradeAction.BUY:
                    trade_action = 'BUY'
                elif action == otrep_core.TradeAction.SELL:
                    trade_action = 'SELL'
                elif action == otrep_core.TradeAction.CLOSE:
                    trade_action = 'CLOSE'
                else:
                    trade_action = 'HOLD'
            else:
                signal, lookback_used, volatility = self.strategy.calculate_signal(df)
                trade_action = self.strategy.get_trade_signal(signal)
                latency_us = self.strategy.last_execution_time_ms * 1000 if hasattr(self.strategy, 'last_execution_time_ms') else 0
            
            self.last_signals[symbol] = signal
            
            # === STEP 5: Log Signal State ===
            vol_str = f" | Vol: {volatility:.4f}" if volatility > 0 else ""
            lb_str = f" | LB: {lookback_used}" if lookback_used != self.config.momentum_lookback else ""
            graph_str = f" | G:{graph_signal:+.2f}" if abs(graph_signal) > 0.01 else ""
            engine_str = "⚡" if self.use_cpp_engine else "🐍"
            latency_str = f" | {latency_us:.2f}µs" if self.use_cpp_engine else ""
            pos_str = f"Pos: {current_qty}" if in_position else "Pos: 0"
            self.log(
                f"{engine_str} {symbol}: ${current_price:.2f} | "
                f"Signal: {signal:+.2f}{graph_str}{lb_str}{vol_str}{latency_str} | {pos_str}"
            )
            
            # === STEP 6: Stop-Loss Check (CRITICAL) ===
            if in_position and position_data:
                position_size = abs(current_qty) * entry_price
                risk_metrics = self.risk_manager.calculate_position_risk(
                    entry_price=entry_price,
                    current_price=current_price,
                    position_size=position_size,
                    symbol=symbol
                )
                
                if risk_metrics['is_stop_triggered']:
                    self.log(f"🚨 EMERGENCY EXIT: Closing {symbol}")
                    try:
                        result = self.exec_client.close_position(symbol)
                        self.log(f"   Order ID: {result.get('id', 'SUBMITTED')}")
                        # Remove from local tracking immediately
                        if symbol in self.open_positions:
                            del self.open_positions[symbol]
                    except Exception as e:
                        self.log(f"   ERROR: {e}")
                    return
            
            # === STEP 6: Trade Entry Logic (BUY/SELL) ===
            qty = int(self.config.position_size_usd / current_price)
            if qty < 1:
                qty = 1
            
            open_count = len(self.open_positions)
            can_open = self.risk_manager.can_open_position(open_count)
            
            if trade_action == 'BUY' and not in_position:
                if not can_open:
                    self.log(f"   SKIPPED: Position limit ({open_count}/{self.config.max_positions})")
                    return
                
                self.log(f"🟢 BUY {symbol}: {qty} @ ${current_price:.2f}")
                try:
                    result = self.exec_client.submit_order(symbol, qty, 'buy')
                    self.log(f"   Order ID: {result.get('id', 'SUBMITTED')}")
                except Exception as e:
                    self.log(f"   ERROR: {e}")
            
            elif trade_action == 'SELL' and not in_position:
                if not can_open:
                    self.log(f"   SKIPPED: Position limit ({open_count}/{self.config.max_positions})")
                    return
                
                self.log(f"🔴 SELL {symbol}: {qty} @ ${current_price:.2f}")
                try:
                    result = self.exec_client.submit_order(symbol, qty, 'sell')
                    self.log(f"   Order ID: {result.get('id', 'SUBMITTED')}")
                except Exception as e:
                    self.log(f"   ERROR: {e}")
            
            elif trade_action == 'CLOSE' and in_position:
                side = 'sell' if current_qty > 0 else 'buy'
                close_qty = abs(current_qty)
                self.log(f"⚪ CLOSE {symbol}: {close_qty} (neutral signal)")
                try:
                    result = self.exec_client.submit_order(symbol, close_qty, side)
                    self.log(f"   Order ID: {result.get('id', 'SUBMITTED')}")
                except Exception as e:
                    self.log(f"   ERROR: {e}")
        
        except Exception as e:
            self.log(f"ERROR processing {symbol}: {e}")
    
    def show_portfolio(self) -> None:
        """Display current portfolio status."""
        try:
            account = self.exec_client.get_account()
            positions = self.exec_client.get_positions()
            
            equity = float(account['equity'])
            cash = float(account['cash'])
            
            daily_pnl = equity - self.risk_manager.starting_equity
            daily_pnl_pct = (daily_pnl / self.risk_manager.starting_equity * 100 
                           if self.risk_manager.starting_equity > 0 else 0)
            
            self.log("=" * 70)
            self.log(f"💰 PORTFOLIO: ${equity:,.2f} | Cash: ${cash:,.2f}")
            self.log(f"📈 Daily P&L: ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)")
            self.log(f"🎯 Positions: {len(positions)}/{self.config.max_positions}")
            
            if positions:
                for pos in positions:
                    sym = pos['symbol']
                    qty = int(pos['qty'])
                    avg = float(pos['avg_entry_price'])
                    cur = float(pos['current_price'])
                    pl = float(pos['unrealized_pl'])
                    pl_pct = float(pos['unrealized_plpc']) * 100
                    
                    stop = avg * (1 - self.config.stop_loss_pct)
                    dist = ((cur - stop) / cur) * 100
                    
                    self.log(
                        f"   {sym}: {qty:+4d} @ ${avg:.2f} → ${cur:.2f} | "
                        f"${pl:+7.2f} ({pl_pct:+.2f}%) | Stop: ${stop:.2f} ({dist:.1f}%)"
                    )
            else:
                self.log("   No open positions")
            
            self.log("=" * 70)
            
        except Exception as e:
            self.log(f"ERROR showing portfolio: {e}")
    
    # =========================================================================
    # HISTORICAL BACKTEST MODE
    # =========================================================================
    
    def run_historical_backtest(self, days_to_run: int = 5) -> Dict:
        """
        Run historical backtest to validate Graph Alpha logic end-to-end.
        
        This method fetches historical 5-minute bars and simulates the trading
        loop exactly as it would run live, allowing validation when market is closed.
        
        Args:
            days_to_run: Number of historical days to simulate (default: 5)
            
        Returns:
            Dict with backtest results (signals, actions, performance)
        """
        self.log("=" * 70)
        self.log("📊 HISTORICAL BACKTEST MODE")
        self.log("=" * 70)
        engine_info = f"C++ ({otrep_core.__version__})" if self.use_cpp_engine else "Python (fallback)"
        self.log(f"   Engine: {engine_info}")
        self.log(f"   Graph Alpha: {'ON' if self.market_graph else 'OFF'}")
        self.log(f"   Days: {days_to_run}")
        self.log(f"   Symbols: {len(self.config.symbols)} tickers")
        self.log(f"   Weights: Mom={self.config.momentum_weight}, MR={self.config.mean_reversion_weight}, Graph={self.config.graph_weight}")
        self.log("=" * 70)
        
        results = {
            'signals': [],
            'actions': [],
            'timestamps': [],
            'graph_edges': [],
            'latencies': [],
        }
        
        # Step 1: Fetch historical data for all symbols
        self.log("\n[1/3] Fetching historical data...")
        
        from datetime import timedelta
        end_date = date.today()
        start_date = end_date - timedelta(days=days_to_run + 3)  # Buffer for weekends
        
        historical_data: Dict[str, List[Dict]] = {}
        symbols_with_data = []
        
        # Use all symbols from config for full backtest
        backtest_symbols = self.config.symbols
        
        for symbol in backtest_symbols:
            try:
                # Request more bars to account for market closures/weekends
                # 78 bars per day (6.5 hours * 12 bars/hour at 5-min timeframe)
                bars_per_day = 78
                requested_bars = bars_per_day * (days_to_run + 2)  # Buffer for weekends
                
                df = self.data_client.get_bars(
                    symbol=symbol,
                    timeframe=self.config.timeframe,
                    limit=max(requested_bars, self.config.graph_lookback_bars * 2)
                )
                if len(df) >= self.config.graph_lookback_bars:
                    # Convert DataFrame to list of bar dicts
                    bars = []
                    for idx, row in df.iterrows():
                        bars.append({
                            'timestamp': idx if isinstance(idx, str) else str(idx),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': int(row['volume'])
                        })
                    historical_data[symbol] = bars
                    symbols_with_data.append(symbol)
                    self.log(f"   ✅ {symbol}: {len(bars)} bars")
            except Exception as e:
                self.log(f"   ❌ {symbol}: {e}")
        
        if len(symbols_with_data) < 3:
            self.log("ERROR: Need at least 3 symbols with data for graph computation")
            return results
        
        self.log(f"\n   Total: {len(symbols_with_data)} symbols with data")
        
        # Step 2: Build aligned timeline
        self.log("\n[2/3] Building aligned timeline...")
        
        # Find common timestamps across all symbols
        min_bars = min(len(historical_data[s]) for s in symbols_with_data)
        self.log(f"   Min bars across symbols: {min_bars}")
        
        # Use the most recent min_bars for all symbols
        for symbol in symbols_with_data:
            historical_data[symbol] = historical_data[symbol][-min_bars:]
        
        # Step 3: Simulate trading loop bar by bar
        self.log("\n[3/3] Running simulation...")
        
        lookback = self.config.graph_lookback_bars
        total_bars = min_bars - lookback
        
        if total_bars <= 0:
            self.log("ERROR: Not enough bars for simulation")
            return results
        
        self.log(f"   Simulating {total_bars} bars (lookback={lookback})...")
        
        # Reset C++ strategies for each symbol
        for symbol in symbols_with_data:
            self._get_cpp_strategy(symbol).clear()
        
        # Initialize strategies with warmup period
        self.log(f"   Warming up strategies with {lookback} bars...")
        for t in range(lookback):
            for symbol in symbols_with_data:
                bar = historical_data[symbol][t]
                cpp_strategy = self._get_cpp_strategy(symbol)
                cpp_bar = otrep_core.Bar(
                    bar['open'], bar['high'], bar['low'], bar['close'], bar['volume']
                )
                cpp_strategy.on_bar(cpp_bar)
        
        # Main simulation loop
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        
        for t in range(lookback, min_bars):
            # Build current price matrix for graph computation
            # Use last 'lookback' bars for returns calculation
            prices = np.zeros((lookback, len(symbols_with_data)))
            for i, symbol in enumerate(symbols_with_data):
                for j in range(lookback):
                    prices[j, i] = historical_data[symbol][t - lookback + j]['close']
            
            # Compute log returns
            returns = np.diff(np.log(prices), axis=0)
            
            # Calculate graph signals
            graph_signals = {}
            graph_edges = 0
            graph_latency = 0
            
            if self.market_graph:
                # --- Pre-Flight Data Diagnostics ---
                # Check shape: need (T-1, N) where T >= 2 and N >= 2
                if returns.ndim != 2:
                    self.log(f"🚨 FATAL: Returns ndim={returns.ndim}, expected 2")
                    continue
                    
                if returns.shape[0] < 2 or returns.shape[1] < 2:
                    self.log(f"🚨 FATAL: Returns shape {returns.shape} too small. Need (T>=2, N>=2)")
                    continue
                
                # Check memory layout: Pybind11/Eigen prefers C_CONTIGUOUS (row-major)
                if not returns.flags['C_CONTIGUOUS']:
                    self.log("⚠️ WARNING: Returns not C_CONTIGUOUS. Forcing copy for C++ compatibility.")
                    returns = np.ascontiguousarray(returns, dtype=np.float64)
                
                # Ensure dtype is float64 for Eigen compatibility
                if returns.dtype != np.float64:
                    returns = returns.astype(np.float64)
                
                # Log diagnostics on first bar only
                if t == lookback:
                    self.log(f"✅ Data Pre-Flight: shape={returns.shape}, "
                            f"C_CONTIGUOUS={returns.flags['C_CONTIGUOUS']}, "
                            f"dtype={returns.dtype}")
                
                # --- MarketGraph C++ Call ---
                try:
                    result = self.market_graph.calculate_signals(returns)
                    graph_edges = result.num_edges
                    graph_latency = result.latency_ns
                    
                    for i, symbol in enumerate(symbols_with_data):
                        if i < len(result.signals):
                            graph_signals[symbol] = float(result.signals[i])
                except Exception as e:
                    self.log(f"🔥 C++ MarketGraph FAILED at t={t}: {e}")
                    self.log(f"   Data: shape={returns.shape}, dtype={returns.dtype}, "
                            f"C_CONTIGUOUS={returns.flags['C_CONTIGUOUS']}")
                    # Re-raise to stop backtest on C++ failure
                    raise
            
            # Update strategies with current bar and get signals
            bar_signals = {}
            bar_actions = {}
            
            for symbol in symbols_with_data:
                bar = historical_data[symbol][t]
                cpp_strategy = self._get_cpp_strategy(symbol)
                
                cpp_bar = otrep_core.Bar(
                    bar['open'], bar['high'], bar['low'], bar['close'], bar['volume']
                )
                cpp_strategy.on_bar(cpp_bar)
                
                # Get graph signal for this symbol
                graph_sig = graph_signals.get(symbol, 0.0)
                
                # Calculate composite signal
                signal_result = cpp_strategy.calculate_signal(graph_sig)
                
                bar_signals[symbol] = signal_result.signal
                
                # Determine action
                if signal_result.signal > self.config.signal_threshold:
                    action = 'BUY'
                    buy_signals += 1
                elif signal_result.signal < -self.config.signal_threshold:
                    action = 'SELL'
                    sell_signals += 1
                else:
                    action = 'HOLD'
                    neutral_signals += 1
                
                bar_actions[symbol] = action
            
            # Record results
            timestamp = historical_data[symbols_with_data[0]][t]['timestamp']
            results['timestamps'].append(timestamp)
            results['signals'].append(bar_signals.copy())
            results['actions'].append(bar_actions.copy())
            results['graph_edges'].append(graph_edges)
            results['latencies'].append(graph_latency)
            
            # Progress log every 50 bars
            if (t - lookback) % 50 == 0:
                self.log(f"   Bar {t - lookback + 1}/{total_bars}: "
                        f"Edges={graph_edges}, "
                        f"Latency={graph_latency/1e6:.2f}ms")
        
        # Summary
        self.log("\n" + "=" * 70)
        self.log("📊 BACKTEST COMPLETE")
        self.log("=" * 70)
        self.log(f"   Total bars simulated: {total_bars}")
        self.log(f"   Symbols: {len(symbols_with_data)}")
        self.log(f"   BUY signals: {buy_signals} ({100*buy_signals/(buy_signals+sell_signals+neutral_signals):.1f}%)")
        self.log(f"   SELL signals: {sell_signals} ({100*sell_signals/(buy_signals+sell_signals+neutral_signals):.1f}%)")
        self.log(f"   HOLD signals: {neutral_signals} ({100*neutral_signals/(buy_signals+sell_signals+neutral_signals):.1f}%)")
        
        avg_edges = np.mean(results['graph_edges']) if results['graph_edges'] else 0
        avg_latency = np.mean(results['latencies']) / 1e6 if results['latencies'] else 0
        self.log(f"   Avg graph edges: {avg_edges:.1f}")
        self.log(f"   Avg graph latency: {avg_latency:.3f} ms")
        
        # Show sample signals from last bar
        if results['signals']:
            self.log("\n   Last bar signals (top 5 by magnitude):")
            last_signals = results['signals'][-1]
            sorted_signals = sorted(last_signals.items(), key=lambda x: abs(x[1]), reverse=True)
            for symbol, sig in sorted_signals[:5]:
                action = results['actions'][-1][symbol]
                self.log(f"      {symbol}: {sig:+.4f} → {action}")
        
        self.log("=" * 70)
        
        return results

    def run(self) -> None:
        """Main trading loop."""
        self.log("=" * 70)
        self.log("🚀 MVT Trader v5.0 Starting (Graph Alpha Engine)")
        self.log("=" * 70)
        engine_info = f"C++ ({otrep_core.get_version()})" if self.use_cpp_engine else "Python (fallback)"
        self.log(f"   Engine: {engine_info}")
        self.log(f"   Graph Alpha: {'ON' if self.market_graph else 'OFF'}")
        self.log(f"   Data source: {'Polygon' if self.data_client.is_available else 'Alpaca'}")
        self.log(f"   Symbols: {len(self.config.symbols)} tickers")
        self.log(f"   Position size: ${self.config.position_size_usd}")
        self.log(f"   Signal threshold: ±{self.config.signal_threshold}")
        self.log(f"   Weights: Mom={self.config.momentum_weight}, MR={self.config.mean_reversion_weight}, Graph={self.config.graph_weight}")
        self.log(f"   Correlation threshold: {self.config.correlation_threshold}")
        self.log(f"   Stop-loss: {self.config.stop_loss_pct:.1%}")
        self.log(f"   Max positions: {self.config.max_positions}")
        self.log(f"   Daily loss limit: {self.config.daily_loss_limit_pct:.1%}")
        self.log(f"   Adaptive lookback: {'ON' if self.config.adaptive_enabled else 'OFF'}")
        self.log("=" * 70)
        
        # Initialize starting equity
        try:
            account = self.exec_client.get_account()
            starting_equity = float(account['equity'])
            self.risk_manager.set_starting_equity(starting_equity)
        except Exception as e:
            self.log(f"WARNING: Could not fetch starting equity: {e}")
            self.risk_manager.set_starting_equity(100000.0)
        
        # Apply correlation filter at startup
        self.active_symbols = self.apply_correlation_filter()
        
        if not self.active_symbols:
            self.log("⚠️  No symbols passed correlation filter - using all symbols")
            self.active_symbols = self.config.symbols.copy()
        
        while True:
            try:
                self.cycle_count += 1
                self.log(f"\n--- Cycle {self.cycle_count} ---")
                
                # === CRITICAL: Sync positions from broker at start of cycle ===
                self.sync_positions()
                
                # === GRAPH ALPHA: Compute graph signals for all symbols ===
                if self.market_graph and self.config.graph_alpha_enabled:
                    self.graph_signals = self.compute_graph_signals()
                else:
                    self.graph_signals = {}
                
                # Daily loss limit check
                try:
                    account = self.exec_client.get_account()
                    current_equity = float(account['equity'])
                    
                    if not self.risk_manager.check_daily_loss_limit(current_equity):
                        self.log("⏸️  Trading halted - waiting...")
                        time.sleep(self.config.check_interval)
                        continue
                        
                except Exception as e:
                    self.log(f"ERROR checking account: {e}")
                    time.sleep(self.config.check_interval)
                    continue
                
                # Refresh correlation filter every 100 cycles
                if self.cycle_count % 100 == 0:
                    self.correlation_cache.clear()
                    self.active_symbols = self.apply_correlation_filter()
                    if not self.active_symbols:
                        self.active_symbols = self.config.symbols.copy()
                
                # Process symbols
                for symbol in self.active_symbols:
                    self.process_symbol(symbol)
                    time.sleep(1)  # Rate limiting
                
                # Portfolio summary
                if self.cycle_count % 10 == 0:
                    self.show_portfolio()
                
                self.log(f"⏳ Waiting {self.config.check_interval}s...")
                time.sleep(self.config.check_interval)
                
            except KeyboardInterrupt:
                self.log("\n🛑 Shutdown requested...")
                break
            except Exception as e:
                self.log(f"ERROR in main loop: {e}")
                time.sleep(10)
        
        self.log("👋 MVT Trader stopped.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Application entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='OTREP-X PRIME MVT Trader v5.0')
    parser.add_argument('--backtest', action='store_true', 
                       help='Run historical backtest instead of live trading')
    parser.add_argument('--days', type=int, default=5,
                       help='Number of days for backtest (default: 5)')
    args = parser.parse_args()
    
    # Load environment
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
    
    # Load config
    config = Config.from_yaml('config.yaml')
    
    # Run trader
    trader = MVTTrader(config)
    
    if args.backtest:
        # Run historical backtest
        results = trader.run_historical_backtest(days_to_run=args.days)
    else:
        # Run live trading
        trader.run()


if __name__ == '__main__':
    main()
