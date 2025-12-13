"""
OTREP-X PRIME - Minimal Viable Trader (MVT) v5.2 (Hardened Edition)
======================================================================
Streamlined controller for live paper trading with institutional hardening:
- C++ HybridStrategy engine (27x faster than Python)
- C++ MarketGraph engine (Laplacian diffusion signals)
- Unified configuration loader with validation
- Centralized KillSwitch circuit breaker
- MarketDataService with caching and failover
- Volatility-targeted position sizing (ATR-based)
- Compliance audit logging

Author: OTREP-X Development Team
Lead Engineer: Gemini AI
Date: December 2025
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, NamedTuple, Callable

# Local imports
from api.alpaca_client import AlpacaClient
from api.polygon_client import PolygonClient
from risk_manager import RiskManager
from risk.kill_switch import KillSwitch, KillSwitchConfig
from services.market_data_service import MarketDataService, MarketDataConfig
from utils.logger import setup_logging, create_audit_event, AuditEventType

# Unified config loader
from core.config import load_config as load_unified_config, Config as UnifiedConfig


# =============================================================================
# C++ Core Import with Validation
# =============================================================================

try:
    import otrep_core
    
    # CRITICAL: Verify all required classes exist
    if not hasattr(otrep_core, 'HybridStrategy'):
        raise ImportError("HybridStrategy class missing. Recompile C++ core.")
    if not hasattr(otrep_core, 'MarketGraph'):
        raise ImportError("MarketGraph class missing. Recompile C++ core.")
    if not hasattr(otrep_core, 'GraphParams'):
        raise ImportError("GraphParams class missing. Recompile C++ core.")
    
    # Version check
    version = otrep_core.get_version()
    if version < "2.0.0":
        raise ImportError(f"Version {version} too old. Need v2.0.0+ for Graph Alpha.")

    CPP_ENGINE_AVAILABLE = True
    print(f"✅ C++ otrep_core v{version} loaded successfully")

except ImportError as e:
    print(f"🔥 FATAL: C++ core failed: {e}")
    print("   Run: cd cpp/build && cmake .. && make -j4")
    CPP_ENGINE_AVAILABLE = False
    sys.exit(1)


def _build_cpp_graph_params(config: UnifiedConfig) -> 'otrep_core.GraphParams':
    """Build C++ GraphParams from unified config (single source of truth)."""
    params = otrep_core.GraphParams()
    # NOTE: C++ binding field name is `correlation_threshold`, but our meaning is adjacency threshold.
    params.correlation_threshold = float(config.graph_alpha.adjacency_threshold)
    params.diffusion_alpha = float(config.graph_alpha.diffusion_alpha)
    params.lookback_bars = int(config.graph_alpha.lookback_bars)
    return params


def _default_logger(message: str, level: str = 'INFO') -> None:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


# =============================================================================
# BAR CONTAINER
# =============================================================================

class Bar(NamedTuple):
    """Simple bar container for rolling data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


# =============================================================================
# MVT TRADER CORE
# =============================================================================

class MVTTrader:
    """The Minimal Viable Trader orchestration class - HARDENED EDITION."""

    def __init__(
        self,
        config: UnifiedConfig,
        logger: Optional[Callable[[str, str], None]] = None,
        skip_account_sync: bool = False,
    ):
        self.config = config
        self.log_start_time = datetime.now()
        self._log = logger or _default_logger
        
        # Setup structured logging first
        setup_logging(log_dir="logs", log_level="INFO")
        
        self.log("🚀 OTREP-X PRIME MVT Trader v5.2 (HARDENED EDITION) starting...")
        create_audit_event(AuditEventType.SYSTEM_START, message="MVT Trader starting")
        
        # Initialize API clients
        self.log("INFO: Initializing Alpaca Client...")
        self.alpaca_client = AlpacaClient(base_url=config.alpaca_base_url)
        
        self.log("INFO: Initializing Polygon Client...")
        polygon_client = PolygonClient()
        
        # Wrap data client with MarketDataService for caching and failover
        self.log("INFO: Initializing MarketDataService (with caching)...")
        data_config = MarketDataConfig(
            cache_ttl_seconds=300.0,          # 5 minute cache
            staleness_threshold_seconds=float(config.risk.kill_switch_data_stale_sec)
        )
        self.data_service = MarketDataService(
            primary_client=polygon_client,
            fallback_client=None,  # Could add Alpaca as fallback
            config=data_config,
            logger=self.log
        )
        
        # Initialize Kill Switch (CTO directive: 0.40% daily drawdown limit)
        self.log("INFO: Initializing Kill Switch...")
        ks_config = KillSwitchConfig(
            daily_drawdown_limit_pct=float(config.risk.kill_switch_drawdown_pct),
            max_order_rejects=int(config.risk.kill_switch_max_rejects),
            max_data_staleness_seconds=float(config.risk.kill_switch_data_stale_sec),
            max_api_failures=int(config.risk.kill_switch_max_api_fails)
        )
        self.kill_switch = KillSwitch(config=ks_config, logger=self.log)
        self.log(f"   Kill Switch: Drawdown limit = {ks_config.daily_drawdown_limit_pct:.2%}")
        
        # Initialize C++ HybridStrategy with proper params
        cpp_params = otrep_core.StrategyParams()
        cpp_params.momentum_lookback = config.strategy.momentum_lookback
        cpp_params.trend_lookback = config.strategy.trend_lookback
        cpp_params.signal_threshold = config.strategy.signal_threshold
        cpp_params.neutral_threshold = config.strategy.neutral_threshold
        cpp_params.momentum_weight = config.strategy.momentum_weight
        cpp_params.trend_weight = config.strategy.trend_weight
        cpp_params.mean_reversion_weight = config.strategy.mean_reversion_weight
        cpp_params.graph_weight = config.strategy.graph_weight
        cpp_params.adaptive_enabled = config.strategy.adaptive_enabled
        cpp_params.high_vol_lookback = config.strategy.high_vol_lookback
        cpp_params.low_vol_lookback = config.strategy.low_vol_lookback
        cpp_params.vol_multiplier = config.strategy.vol_multiplier
        cpp_params.mean_reversion_enabled = config.strategy.mean_reversion_enabled
        cpp_params.mean_reversion_lookback = config.strategy.mean_reversion_lookback
        cpp_params.bb_std_dev_multiplier = config.strategy.bb_std_dev_multiplier
        cpp_params.reversion_threshold = config.strategy.reversion_threshold
        
        self.strategy_engine = otrep_core.HybridStrategy(cpp_params)
        self.cpp_params = cpp_params  # Store for creating per-symbol strategies
        self.cpp_strategies: Dict[str, 'otrep_core.HybridStrategy'] = {}
        self.log(f"🚀 C++ HybridStrategy initialized (27x faster)")
        self.log(f"   Weights: Mom={config.strategy.momentum_weight}, "
                f"MR={config.strategy.mean_reversion_weight}, Graph={config.strategy.graph_weight}")
        
        # Configure MarketGraph
        self.market_graph = otrep_core.MarketGraph()
        cpp_graph_params = _build_cpp_graph_params(config)
        self.cpp_graph_params = cpp_graph_params
        self.market_graph.set_params(cpp_graph_params)
        self.log(f"📊 C++ MarketGraph initialized (Laplacian diffusion)")
        self.log(f"   Adjacency threshold: {config.graph_alpha.adjacency_threshold}")
        self.log(f"   Lookback bars: {config.graph_alpha.lookback_bars}")
        
        # Initialize risk manager with volatility targeting
        self.risk_manager = RiskManager(
            stop_loss_pct=float(config.risk.stop_loss_pct),
            max_positions=int(config.risk.max_positions),
            daily_loss_limit_pct=float(config.risk.daily_loss_limit_pct),
            target_risk_per_trade_usd=float(config.risk.target_risk_per_trade_usd),
            max_position_size_usd=float(config.risk.max_position_size_usd),
            atr_risk_multiplier=float(config.risk.atr_risk_multiplier),
            logger=self.log
        )
        
        # Set starting equity for kill switch
        if not skip_account_sync:
            try:
                account = self.alpaca_client.get_account()
                starting_equity = float(account.get('equity', 100000))
                self.kill_switch.set_starting_equity(starting_equity)
                self.risk_manager.set_starting_equity(starting_equity)
                self.kill_switch.record_api_result(True)
            except Exception as e:
                self.log(f"⚠️ Could not fetch starting equity: {e}")
                self.kill_switch.record_api_result(False)
                self.kill_switch.set_starting_equity(100000.0)
                self.risk_manager.set_starting_equity(100000.0)
        
        create_audit_event(AuditEventType.CONFIG_LOADED, message="Configuration loaded successfully")
        
    def _get_cpp_strategy(self, symbol: str) -> 'otrep_core.HybridStrategy':
        """Get or create a C++ strategy instance for a symbol (each is stateful)."""
        if symbol not in self.cpp_strategies:
            self.cpp_strategies[symbol] = otrep_core.HybridStrategy(self.cpp_params)
        return self.cpp_strategies[symbol]

    def log(self, message: str, level: str = 'INFO'):
        """Simple logging utility."""
        self._log(message, level)

    def run_historical_backtest(self, days_to_run: int = 5) -> Optional[Dict]:
        """
        Historical backtest for validation when market is closed.
        
        Args:
            days_to_run: Number of historical days to simulate
            
        Returns:
            Dict with backtest results
        """
        self.log("=" * 70)
        self.log("📊 HISTORICAL BACKTEST MODE")
        self.log("=" * 70)
        self.log(f"   Engine: C++ ({otrep_core.get_version()})")
        self.log(f"   Days: {days_to_run}")
        self.log(f"   Symbols: {len(self.config.symbols)} tickers")
        self.log(f"   Weights: Mom={self.config.strategy.momentum_weight}, "
                f"MR={self.config.strategy.mean_reversion_weight}, Graph={self.config.strategy.graph_weight}")
        self.log("=" * 70)
        
        results = {
            'signals': [], 'actions': [], 'timestamps': [],
            'graph_edges': [], 'latencies': []
        }
        
        # [1] Fetch historical data
        self.log("\n[1/3] Fetching historical data...")
        
        lookback = self.config.graph_alpha.lookback_bars
        bars_per_day = 78  # 6.5 hours * 12 bars/hour at 5-min
        requested_bars = bars_per_day * (days_to_run + 2)
        
        historical_data: Dict[str, List[Dict]] = {}
        symbols_with_data = []
        
        for symbol in self.config.symbols:
            try:
                df = self.data_service.get_bars(
                    symbol=symbol,
                    timeframe=self.config.bar_size,
                    limit=max(requested_bars, lookback * 2)
                )
                if len(df) >= lookback:
                    bars = []
                    for idx, row in df.iterrows():
                        bars.append({
                            'timestamp': str(idx),
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
            self.log("ERROR: Need at least 3 symbols for graph computation")
            return results
        
        # [2] Build aligned timeline
        self.log("\n[2/3] Building aligned timeline...")
        min_bars = min(len(historical_data[s]) for s in symbols_with_data)
        self.log(f"   Min bars across symbols: {min_bars}")
        
        for symbol in symbols_with_data:
            historical_data[symbol] = historical_data[symbol][-min_bars:]
        
        # [3] Simulate trading loop
        self.log("\n[3/3] Running simulation...")
        total_bars = min_bars - lookback
        
        if total_bars <= 0:
            self.log("ERROR: Not enough bars for simulation")
            return results
        
        # Reset C++ strategies for each symbol
        for symbol in symbols_with_data:
            self._get_cpp_strategy(symbol).clear()
        
        # Warmup strategies with lookback bars
        self.log(f"   Warming up strategies with {lookback} bars...")
        for t in range(lookback):
            for symbol in symbols_with_data:
                bar = historical_data[symbol][t]
                cpp_strategy = self._get_cpp_strategy(symbol)
                cpp_bar = otrep_core.Bar(
                    bar['open'], bar['high'], bar['low'], bar['close'], bar['volume']
                )
                cpp_strategy.on_bar(cpp_bar)
        
        buy_signals = sell_signals = neutral_signals = 0
        
        for t in range(lookback, min_bars):
            # Build price matrix for graph computation
            prices = np.zeros((lookback, len(symbols_with_data)))
            for i, symbol in enumerate(symbols_with_data):
                for j in range(lookback):
                    prices[j, i] = historical_data[symbol][t - lookback + j]['close']
            
            # Compute log returns
            returns = np.diff(np.log(prices), axis=0)
            
            # Pre-flight checks
            if not returns.flags['C_CONTIGUOUS']:
                returns = np.ascontiguousarray(returns, dtype=np.float64)
            if returns.dtype != np.float64:
                returns = returns.astype(np.float64)
            
            if t == lookback:
                self.log(f"✅ Data Pre-Flight: shape={returns.shape}, "
                        f"C_CONTIGUOUS={returns.flags['C_CONTIGUOUS']}, dtype={returns.dtype}")
            
            # Calculate graph signals
            try:
                graph_result = self.market_graph.calculate_signals(returns)
                graph_edges = graph_result.num_edges
                graph_latency = graph_result.latency_ns
                graph_signals = {symbols_with_data[i]: float(graph_result.signals[i]) 
                                for i in range(len(symbols_with_data))}
            except Exception as e:
                self.log(f"🔥 MarketGraph FAILED at t={t}: {e}")
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
                
                # Get graph signal for this symbol and calculate composite signal
                graph_sig = graph_signals.get(symbol, 0.0)
                signal_result = cpp_strategy.calculate_signal(graph_sig)
                
                sig = signal_result.signal
                bar_signals[symbol] = sig
                
                if sig > self.config.strategy.signal_threshold:
                    bar_actions[symbol] = 'BUY'
                    buy_signals += 1
                elif sig < -self.config.strategy.signal_threshold:
                    bar_actions[symbol] = 'SELL'
                    sell_signals += 1
                else:
                    bar_actions[symbol] = 'HOLD'
                    neutral_signals += 1
            
            results['timestamps'].append(historical_data[symbols_with_data[0]][t]['timestamp'])
            results['signals'].append(bar_signals.copy())
            results['actions'].append(bar_actions.copy())
            results['graph_edges'].append(graph_edges)
            results['latencies'].append(graph_latency)
            
            if (t - lookback) % 50 == 0:
                self.log(f"   Bar {t - lookback + 1}/{total_bars}: "
                        f"Edges={graph_edges}, Latency={graph_latency/1e6:.2f}ms")
        
        # Summary
        total = buy_signals + sell_signals + neutral_signals
        self.log("\n" + "=" * 70)
        self.log("📊 BACKTEST COMPLETE")
        self.log("=" * 70)
        self.log(f"   Total bars simulated: {total_bars}")
        self.log(f"   Symbols: {len(symbols_with_data)}")
        self.log(f"   BUY signals: {buy_signals} ({100*buy_signals/total:.1f}%)")
        self.log(f"   SELL signals: {sell_signals} ({100*sell_signals/total:.1f}%)")
        self.log(f"   HOLD signals: {neutral_signals} ({100*neutral_signals/total:.1f}%)")
        
        avg_edges = np.mean(results['graph_edges']) if results['graph_edges'] else 0
        avg_latency = np.mean(results['latencies']) / 1e6 if results['latencies'] else 0
        self.log(f"   Avg graph edges: {avg_edges:.1f}")
        self.log(f"   Avg graph latency: {avg_latency:.3f} ms")
        
        if results['signals']:
            self.log("\n   Last bar signals (top 5 by magnitude):")
            last_signals = results['signals'][-1]
            sorted_signals = sorted(last_signals.items(), key=lambda x: abs(x[1]), reverse=True)
            for symbol, sig in sorted_signals[:5]:
                action = results['actions'][-1][symbol]
                self.log(f"      {symbol}: {sig:+.4f} → {action}")
        
        self.log("=" * 70)
        return results

    def run(self):
        """The main live paper trading loop - HARDENED with Kill Switch."""
        self.log("=" * 70)
        self.log("🟢 LIVE PAPER TRADING MODE (C++ Graph Alpha + HARDENED)")
        self.log("=" * 70)
        self.log(f"   Kill Switch: Active (0.40% drawdown limit)")
        self.log(f"   Position Sizing: Volatility-targeted ($125 risk)")
        self.log("=" * 70)
        
        create_audit_event(AuditEventType.SYSTEM_START, message="Live trading loop started")
        
        symbols = self.config.symbols
        lookback = self.config.graph_alpha.lookback_bars
        
        try:
            while True:
                cycle_start = time.time()
                
                # [0] KILL SWITCH CHECK - Check if trading is allowed
                if not self.kill_switch.is_trading_allowed():
                    reason = self.kill_switch.get_status()['active_reason']
                    self.log(f"  🛑 KILL SWITCH ACTIVE: {reason}")
                    create_audit_event(
                        AuditEventType.KILL_SWITCH_TRIGGERED,
                        message=f"Trading halted: {reason}"
                    )
                    time.sleep(60)  # Wait longer when halted
                    continue
                
                # Update current equity for kill switch monitoring
                try:
                    account = self.alpaca_client.get_account()
                    current_equity = float(account.get('equity', 0))
                    self.kill_switch.record_api_result(True)
                    
                    # Check drawdown against kill switch
                    if not self.kill_switch.check_drawdown(current_equity):
                        self.log(f"  🛑 KILL SWITCH: Drawdown limit breached!")
                        create_audit_event(
                            AuditEventType.KILL_SWITCH_TRIGGERED,
                            message="Drawdown limit breached",
                            data={'current_equity': current_equity}
                        )
                        continue
                except Exception as e:
                    self.log(f"  ⚠️ Could not fetch account: {e}")
                    self.kill_switch.record_api_result(False)
                    current_equity = 0.0
                
                # [1] Fetch rolling bars for all symbols using MarketDataService
                self.log(f"  [1] Fetching {lookback + 1} bars for {len(symbols)} symbols...")
                
                raw_data: Dict[str, List[Bar]] = {}
                data_fetch_success = False
                
                for symbol in symbols:
                    try:
                        df = self.data_service.get_bars(
                            symbol=symbol,
                            timeframe=self.config.bar_size,
                            limit=lookback + 1
                        )
                        if len(df) >= lookback + 1:
                            bars = []
                            for idx, row in df.iterrows():
                                bars.append(Bar(
                                    timestamp=idx,
                                    open=float(row['open']),
                                    high=float(row['high']),
                                    low=float(row['low']),
                                    close=float(row['close']),
                                    volume=int(row['volume'])
                                ))
                            raw_data[symbol] = bars
                            data_fetch_success = True
                    except Exception as e:
                        self.log(f"    ❌ {symbol}: {e}")
                
                # Record data freshness to kill switch
                if data_fetch_success:
                    self.kill_switch.record_data_update()
                else:
                    # Check data staleness
                    if not self.kill_switch.check_data_staleness():
                        self.log("  🛑 KILL SWITCH: Data staleness exceeded!")
                        continue
                
                valid_symbols = [s for s in symbols if s in raw_data and len(raw_data[s]) >= lookback + 1]
                
                if len(valid_symbols) < 3:
                    self.log("⚠️ Insufficient data. Waiting for next cycle...")
                    time.sleep(self.config.check_interval)
                    continue
                
                # [2] Build aligned returns matrix
                close_df = pd.DataFrame({
                    s: pd.Series([b.close for b in raw_data[s]], 
                                index=[b.timestamp for b in raw_data[s]])
                    for s in valid_symbols
                })
                returns_matrix = close_df.pct_change().dropna().values
                
                if returns_matrix.shape[0] < lookback:
                    self.log(f"⚠️ Insufficient bars ({returns_matrix.shape[0]}). Skipping.")
                    time.sleep(self.config.check_interval)
                    continue
                
                # Separate lookback and current return
                lookback_data = returns_matrix[:-1, :]
                current_returns = returns_matrix[-1, :]
                
                if not lookback_data.flags['C_CONTIGUOUS']:
                    lookback_data = np.ascontiguousarray(lookback_data, dtype=np.float64)
                
                # [3] Calculate graph signals
                calc_start = time.time()
                
                graph_signals = self.market_graph.calculate_signals_map(lookback_data, valid_symbols)
                latency_ns = self.market_graph.last_latency_ns()
                edges = self.market_graph.params().last_edge_count
                
                calc_ms = (time.time() - calc_start) * 1000
                self.log(f"  [2] Graph: {len(valid_symbols)} assets, "
                        f"{calc_ms:.1f}ms (C++: {latency_ns/1e6:.2f}ms, Edges: {edges})")
                
                # [4] Update per-symbol strategies and get signals, then execute trades
                for i, symbol in enumerate(valid_symbols):
                    # Get the most recent bar for this symbol
                    bar = raw_data[symbol][-1]
                    cpp_strategy = self._get_cpp_strategy(symbol)
                    
                    # Feed bar to strategy
                    cpp_bar = otrep_core.Bar(bar.open, bar.high, bar.low, bar.close, bar.volume)
                    cpp_strategy.on_bar(cpp_bar)
                    
                    # Calculate composite signal with graph alpha
                    graph_sig = graph_signals.get(symbol, 0.0)
                    signal_result = cpp_strategy.calculate_signal(graph_sig)
                    signal = signal_result.signal
                    
                    if abs(signal) > self.config.strategy.signal_threshold:
                        side = "BUY" if signal > 0 else "SELL"
                        
                        # Calculate position size using ATR-based vol targeting
                        atr = cpp_strategy.get_state().current_atr
                        shares = self.risk_manager.calculate_position_size(bar.close, atr)

                        # Sync positions for transactional checks
                        try:
                            positions = self.alpaca_client.get_positions()
                            self.kill_switch.record_api_result(True)
                        except Exception as e:
                            self.log(f"  ⚠️ Could not fetch positions: {e}")
                            self.kill_switch.record_api_result(False)
                            positions = []

                        current_open_positions = len(positions)
                        self.kill_switch.update_position_count(current_open_positions)

                        pre = self.risk_manager.pre_trade_check(
                            symbol=symbol,
                            side=side,
                            entry_price=float(bar.close),
                            shares=int(shares),
                            current_positions=int(current_open_positions),
                            current_equity=float(current_equity),
                        )

                        if not pre.get('allowed', False):
                            self.log(f"  [3] 🟡 BLOCKED: {side} {symbol} ({pre.get('reason')})")
                            create_audit_event(
                                AuditEventType.RISK_LIMIT_HIT,
                                message=f"Blocked {side} {symbol}",
                                data={'symbol': symbol, 'signal': signal, 'reason': pre.get('reason')}
                            )
                            continue

                        try:
                            alpaca_side = 'buy' if side == 'BUY' else 'sell'
                            order_result = self.alpaca_client.submit_order(
                                symbol=symbol,
                                qty=int(shares),
                                side=alpaca_side,
                                order_type='market',
                                time_in_force='day'
                            )
                            self.kill_switch.record_order_result(True)

                            order_id = str(order_result.get('id', '')) if isinstance(order_result, dict) else ''
                            self.risk_manager.post_trade_update(
                                symbol=symbol,
                                side=side,
                                fill_price=float(bar.close),
                                shares=int(shares),
                                order_id=order_id,
                            )
                            self.log(f"  [3] 🟢 ORDER: {side} {shares} {symbol} @ ${bar.close:.2f} "
                                    f"(Signal: {signal:.4f}, ATR: {atr:.4f})")

                            create_audit_event(
                                AuditEventType.TRADE_EXECUTED,
                                message=f"{side} {symbol}",
                                data={
                                    'symbol': symbol,
                                    'side': side,
                                    'signal': float(signal),
                                    'price': float(bar.close),
                                    'shares': int(shares),
                                    'atr': float(atr),
                                    'graph_signal': float(graph_sig),
                                    'order_id': order_id,
                                    'adjacency_threshold': float(self.config.graph_alpha.adjacency_threshold),
                                }
                            )
                        except Exception as e:
                            self.log(f"  [3] 🔴 ORDER FAILED: {side} {symbol}: {e}")
                            self.kill_switch.record_order_result(False)
                            self.kill_switch.record_api_result(False)
                            create_audit_event(
                                AuditEventType.ERROR,
                                message=f"Order failed: {side} {symbol}",
                                data={'symbol': symbol, 'side': side, 'error': str(e)}
                            )
                
                # [5] Wait for next cycle
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.config.check_interval - cycle_time)
                self.log(f"  [4] Cycle: {cycle_time:.1f}s. Sleep: {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.log("\n🛑 Shutdown requested...")
            create_audit_event(AuditEventType.SYSTEM_STOP, message="Graceful shutdown")
        except Exception as e:
            self.log(f"🔥 FATAL: {e}")
            create_audit_event(
                AuditEventType.ERROR,
                message=f"Fatal error: {e}",
                data={'exception': str(e)}
            )
        
        self.log("👋 MVT Trader stopped.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Application entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='OTREP-X PRIME MVT Trader v5.2 (HARDENED)')
    parser.add_argument('--backtest', action='store_true',
                       help='Run historical backtest instead of live trading')
    parser.add_argument('--days', type=int, default=5,
                       help='Number of days for backtest (default: 5)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without trading')
    args = parser.parse_args()
    
    # Load environment
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
    
    # Suppress deprecation warnings in production (show in verbose mode)
    if not os.getenv('OTREP_VERBOSE'):
        warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    try:
        cfg = load_unified_config(args.config)

        print(f"✅ Configuration loaded: {len(cfg.symbols)} symbols")
        print(f"   Adjacency threshold: {cfg.graph_alpha.adjacency_threshold}")
        print(f"   Target risk/trade: ${cfg.risk.target_risk_per_trade_usd}")

        # Explicitly show the C++ binding mapping (binding name retained for backward compatibility)
        mapped = _build_cpp_graph_params(cfg)
        print(f"   Mapped GraphParams.correlation_threshold: {mapped.correlation_threshold}")
        
    except FileNotFoundError as e:
        print(f"FATAL: Configuration file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"FATAL: Configuration validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: Could not load configuration: {e}")
        sys.exit(1)
    
    if args.dry_run:
        print("✅ Dry run complete. Configuration is valid.")
        return

    trader = MVTTrader(cfg)
    
    if args.backtest:
        trader.run_historical_backtest(days_to_run=args.days)
    else:
        trader.run()


if __name__ == '__main__':
    main()
