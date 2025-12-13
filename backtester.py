"""
Backtester Framework with Parameter Optimization
=================================================
Full backtesting and walk-forward optimization for HybridStrategy.

Author: OTREP-X Development Team
Lead Engineer: Gemini AI
Phase III Implementation
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from itertools import product
import pandas as pd
import numpy as np

try:
    import backtrader as bt
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False

from strategy.momentum import HybridStrategy


# =============================================================================
# SIMPLE BACKTESTER
# =============================================================================

class SimpleBacktester:
    """
    Simple backtesting framework for HybridStrategy.
    
    Includes realistic cost modeling:
    - Slippage (basis points)
    - Fixed commission per trade
    - Percentage-based commission
    """
    
    def __init__(
        self,
        strategy: HybridStrategy,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.10,
        stop_loss_pct: float = 0.02,
        commission_pct: float = 0.001,
        slippage_bps: float = 5.0,
        commission_fixed: float = 1.0
    ):
        """
        Initialize simple backtester with cost modeling.
        
        Args:
            strategy: HybridStrategy instance
            initial_capital: Starting capital
            position_size_pct: Position size as percentage of capital
            stop_loss_pct: Stop-loss percentage
            commission_pct: Commission per trade (percentage)
            slippage_bps: Slippage in basis points (e.g., 5.0 = 0.05%)
            commission_fixed: Fixed commission per trade in dollars
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.commission_pct = commission_pct
        self.slippage_bps = slippage_bps
        self.commission_fixed = commission_fixed
    
    def run(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV data (indexed by datetime)
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dict with performance metrics
        """
        # Filter by date range if provided
        data = df.copy()
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if len(data) < 30:
            return self._empty_results()
        
        # Generate signals for all data
        signals_df = self.strategy.generate_signals(data)
        
        # Trading simulation
        capital = self.initial_capital
        position = 0
        entry_price = 0.0
        trades: List[Dict] = []
        equity_curve = []
        
        # Slippage factor: convert basis points to decimal
        slippage_factor = self.slippage_bps / 10000.0
        
        for i, (timestamp, row) in enumerate(signals_df.iterrows()):
            current_price = row['close']
            signal = row['signal']
            
            # Track equity
            position_value = position * current_price if position > 0 else 0
            total_equity = capital + position_value
            equity_curve.append({
                'timestamp': timestamp,
                'equity': total_equity,
                'position': position
            })
            
            # Check stop-loss
            if position > 0 and entry_price > 0:
                if (current_price - entry_price) / entry_price <= -self.stop_loss_pct:
                    # SELL with slippage: execute at lower price
                    executed_price = current_price * (1 - slippage_factor)
                    sale_value = position * executed_price * (1 - self.commission_pct) - self.commission_fixed
                    pnl = sale_value - (position * entry_price)
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'STOP_LOSS',
                        'price': executed_price,
                        'shares': position,
                        'pnl': pnl
                    })
                    capital += sale_value
                    position = 0
                    entry_price = 0.0
                    continue
            
            # Process signals
            trade_action = self.strategy.get_trade_signal(signal)
            
            if trade_action == 'BUY' and position == 0:
                # BUY with slippage: execute at higher price
                executed_price = current_price * (1 + slippage_factor)
                position_value = capital * self.position_size_pct
                shares = int(position_value / executed_price)
                
                if shares > 0:
                    cost = shares * executed_price * (1 + self.commission_pct) + self.commission_fixed
                    if cost <= capital:
                        capital -= cost
                        position = shares
                        entry_price = executed_price
                        trades.append({
                            'timestamp': timestamp,
                            'type': 'BUY',
                            'price': executed_price,
                            'shares': shares,
                            'pnl': 0
                        })
            
            elif trade_action == 'SELL' and position > 0:
                # SELL with slippage: execute at lower price
                executed_price = current_price * (1 - slippage_factor)
                sale_value = position * executed_price * (1 - self.commission_pct) - self.commission_fixed
                pnl = sale_value - (position * entry_price)
                trades.append({
                    'timestamp': timestamp,
                    'type': 'SELL',
                    'price': executed_price,
                    'shares': position,
                    'pnl': pnl
                })
                capital += sale_value
                position = 0
                entry_price = 0.0
            
            elif trade_action == 'CLOSE' and position > 0:
                # CLOSE with slippage: execute at lower price
                executed_price = current_price * (1 - slippage_factor)
                sale_value = position * executed_price * (1 - self.commission_pct) - self.commission_fixed
                pnl = sale_value - (position * entry_price)
                trades.append({
                    'timestamp': timestamp,
                    'type': 'CLOSE',
                    'price': current_price,
                    'shares': position,
                    'pnl': pnl
                })
                capital += sale_value
                position = 0
                entry_price = 0.0
        
        return self._calculate_metrics(
            capital, position, data, equity_curve, trades
        )
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results when insufficient data."""
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.initial_capital,
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'equity_curve': pd.DataFrame(),
            'trades': pd.DataFrame()
        }
    
    def _calculate_metrics(
        self,
        capital: float,
        position: int,
        data: pd.DataFrame,
        equity_curve: List[Dict],
        trades: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate performance metrics."""
        final_equity = capital + (
            position * data['close'].iloc[-1] if position > 0 else 0
        )
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        equity_df = pd.DataFrame(equity_curve)
        if len(equity_df) > 0:
            equity_df.set_index('timestamp', inplace=True)
        
        # Sharpe ratio
        if len(equity_df) > 1:
            returns = equity_df['equity'].pct_change().dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        if len(equity_df) > 0:
            rolling_max = equity_df['equity'].cummax()
            drawdowns = (equity_df['equity'] - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
        else:
            max_drawdown = 0
        
        # Trade statistics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
        else:
            winning_trades = pd.DataFrame()
            losing_trades = pd.DataFrame()
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'equity_curve': equity_df,
            'trades': trades_df
        }
    
    def print_report(self, results: Dict) -> None:
        """Print formatted backtest report."""
        print("=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Initial Capital:    ${results['initial_capital']:,.2f}")
        print(f"Final Equity:       ${results['final_equity']:,.2f}")
        print(f"Total Return:       {results['total_return_pct']:+.2f}%")
        print(f"Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:       {results['max_drawdown_pct']:.2f}%")
        print("-" * 60)
        print(f"Total Trades:       {results['total_trades']}")
        print(f"Winning Trades:     {results['winning_trades']}")
        print(f"Losing Trades:      {results['losing_trades']}")
        print(f"Win Rate:           {results['win_rate']*100:.1f}%")
        print(f"Avg Win:            ${results['avg_win']:,.2f}")
        print(f"Avg Loss:           ${results['avg_loss']:,.2f}")
        print("=" * 60)


# =============================================================================
# PARAMETER OPTIMIZER
# =============================================================================

class ParameterOptimizer:
    """
    Walk-forward parameter optimization for HybridStrategy.
    
    Tests all combinations of parameters and returns the best
    configuration based on Sharpe Ratio.
    
    Phase IV: Includes realistic cost modeling and latency profiling.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.10,
        stop_loss_pct: float = 0.02,
        commission_pct: float = 0.001,
        slippage_bps: float = 5.0,
        commission_fixed: float = 1.0
    ):
        """
        Initialize optimizer with cost modeling.
        
        Args:
            initial_capital: Starting capital for backtests
            position_size_pct: Position size percentage
            stop_loss_pct: Stop-loss percentage
            commission_pct: Commission per trade (percentage)
            slippage_bps: Slippage in basis points
            commission_fixed: Fixed commission per trade in dollars
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.commission_pct = commission_pct
        self.slippage_bps = slippage_bps
        self.commission_fixed = commission_fixed
    
    def optimize_strategy(
        self,
        data: pd.DataFrame,
        optim_params: Dict[str, List[Any]],
        base_params: Optional[Dict[str, Any]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data: DataFrame with OHLCV data
            optim_params: Dict mapping parameter names to lists of values
            base_params: Base parameters to use (non-optimized params)
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dict with best parameters, performance metrics, and latency stats
        """
        base_params = base_params or {}
        
        # Generate all parameter combinations
        param_names = list(optim_params.keys())
        param_values = list(optim_params.values())
        combinations = list(product(*param_values))
        
        print(f"🔍 Testing {len(combinations)} parameter combinations...")
        print(f"   💰 Cost Model: Slippage={self.slippage_bps}bps, Commission=${self.commission_fixed}/trade")
        
        best_sharpe = float('-inf')
        best_params = {}
        best_results = None
        best_strategy = None
        all_results = []
        all_latencies = []
        
        for i, combo in enumerate(combinations):
            # Build parameter dict for this combination
            params = base_params.copy()
            for name, value in zip(param_names, combo):
                params[name] = value
            
            # Create strategy with these parameters
            strategy = HybridStrategy(**params)
            strategy.reset_profiling()  # Reset latency tracking
            
            # Run backtest with cost modeling
            backtester = SimpleBacktester(
                strategy=strategy,
                initial_capital=self.initial_capital,
                position_size_pct=self.position_size_pct,
                stop_loss_pct=self.stop_loss_pct,
                commission_pct=self.commission_pct,
                slippage_bps=self.slippage_bps,
                commission_fixed=self.commission_fixed
            )
            
            results = backtester.run(data, start_date, end_date)
            sharpe = results['sharpe_ratio']
            
            # Track latency from strategy
            avg_latency = strategy.avg_execution_time_ms
            all_latencies.append(avg_latency)
            
            # Track result
            result_entry = {
                'params': params.copy(),
                'sharpe_ratio': sharpe,
                'total_return': results['total_return_pct'],
                'max_drawdown': results['max_drawdown_pct'],
                'total_trades': results['total_trades'],
                'avg_latency_ms': avg_latency
            }
            all_results.append(result_entry)
            
            # Update best if improved
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params.copy()
                best_results = results
                best_strategy = strategy
            
            # Progress indicator
            if (i + 1) % 10 == 0 or i == len(combinations) - 1:
                print(f"   Tested {i + 1}/{len(combinations)} combinations...")
        
        # Calculate overall average latency
        avg_latency_all = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
        
        return {
            'best_params': best_params,
            'best_sharpe': best_sharpe,
            'best_results': best_results,
            'best_strategy': best_strategy,
            'all_results': all_results,
            'combinations_tested': len(combinations),
            'avg_latency_ms': avg_latency_all,
            'cost_model': {
                'slippage_bps': self.slippage_bps,
                'commission_fixed': self.commission_fixed
            }
        }
    
    def print_optimization_report(self, opt_result: Dict) -> None:
        """Print formatted optimization report with Phase IV metrics."""
        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS (PHASE IV - WITH COSTS)")
        print("=" * 60)
        print(f"Combinations Tested: {opt_result['combinations_tested']}")
        
        # Cost model info
        if 'cost_model' in opt_result:
            print(f"\n💰 Cost Model Applied:")
            print(f"   Slippage:   {opt_result['cost_model']['slippage_bps']} bps")
            print(f"   Commission: ${opt_result['cost_model']['commission_fixed']}/trade")
        
        print(f"\n📊 Best Sharpe Ratio (Post-Costs): {opt_result['best_sharpe']:.4f}")
        
        # Latency info
        if 'avg_latency_ms' in opt_result:
            print(f"⚡ Average Latency: {opt_result['avg_latency_ms']:.4f} ms/bar")
        
        print("\n🎯 Optimal Parameters:")
        for key, value in opt_result['best_params'].items():
            print(f"   {key}: {value}")
        
        if opt_result['best_results']:
            print("\n📈 Best Backtest Results:")
            print(f"   Total Return:  {opt_result['best_results']['total_return_pct']:+.2f}%")
            print(f"   Max Drawdown:  {opt_result['best_results']['max_drawdown_pct']:.2f}%")
            print(f"   Total Trades:  {opt_result['best_results']['total_trades']}")
            print(f"   Win Rate:      {opt_result['best_results']['win_rate']*100:.1f}%")
        print("=" * 60)


# =============================================================================
# WALK-FORWARD OPTIMIZATION
# =============================================================================

def run_optimization_walk_forward(
    data: Optional[pd.DataFrame] = None,
    symbol: str = 'SPY',
    days: int = 180,
    slippage_bps: float = 5.0,
    commission_fixed: float = 1.0,
    initial_capital: float = 100000.0
) -> Dict[str, Any]:
    """
    Run walk-forward optimization on historical data with realistic costs.
    
    Phase IV: High-Fidelity Validation with cost modeling and latency profiling.
    
    If no data provided, attempts to fetch from PolygonClient.
    
    Args:
        data: Optional DataFrame with OHLCV data
        symbol: Symbol to fetch data for (if data not provided)
        days: Days of historical data to fetch
        slippage_bps: Slippage in basis points (default: 5.0)
        commission_fixed: Fixed commission per trade (default: $1.00)
        initial_capital: Starting capital (default: $100,000)
        
    Returns:
        Dict with optimization results including latency metrics
    """
    print("=" * 60)
    print("🚀 PHASE IV: High-Fidelity Validation & Performance Profiling")
    print("=" * 60)
    
    # Fetch data if not provided
    if data is None:
        try:
            from api.polygon_client import PolygonClient
            client = PolygonClient()
            print(f"📊 Fetching {days} days of REAL {symbol} data...")
            data = client.get_daily_bars(symbol, days)
            print(f"   ✅ Retrieved {len(data)} bars")
            
            # Fall back to synthetic if insufficient data from API
            if len(data) < 50:
                print("   ⚠️ Insufficient API data, using synthetic data for demonstration...")
                data = _generate_synthetic_data(days)
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            print("   Using synthetic data for demonstration...")
            data = _generate_synthetic_data(days)
    
    if len(data) < 50:
        print("❌ Insufficient data for optimization")
        return {}
    
    # Print cost model
    print(f"\n💰 Cost Model (The Quant's Tax):")
    print(f"   Slippage:   {slippage_bps} basis points")
    print(f"   Commission: ${commission_fixed:.2f} per trade")
    print(f"   Capital:    ${initial_capital:,.0f}")
    
    # Define optimization ranges per Gemini's Phase III/IV instructions
    optim_params = {
        'momentum_lookback': [10, 20, 30],
        'mean_reversion_lookback': [15, 20, 25],
        'bb_std_dev_multiplier': [1.5, 2.0, 2.5]
    }
    
    # Base parameters (non-optimized)
    base_params = {
        'signal_threshold': 0.15,
        'neutral_threshold': 0.05,
        'momentum_weight': 0.5,
        'mean_reversion_weight': 0.5,
        'mean_reversion_enabled': True,
        'mean_reversion_lookback': 20,
        'adaptive_enabled': True,
        'high_vol_lookback': 10,
        'low_vol_lookback': 30,
        'vol_multiplier': 1.5
    }
    
    print(f"\n📈 Optimization Range:")
    for param, values in optim_params.items():
        print(f"   {param}: {values}")
    
    # Run optimization with cost modeling
    optimizer = ParameterOptimizer(
        initial_capital=initial_capital,
        position_size_pct=0.10,
        stop_loss_pct=0.02,
        slippage_bps=slippage_bps,
        commission_fixed=commission_fixed
    )
    
    result = optimizer.optimize_strategy(
        data=data,
        optim_params=optim_params,
        base_params=base_params
    )
    
    # Print results
    optimizer.print_optimization_report(result)
    
    print("\n" + "=" * 60)
    print("✅ Phase IV Complete. Optimal Parameters Found:")
    print("=" * 60)
    for key, value in result['best_params'].items():
        if key in optim_params:
            print(f"   🎯 {key}: {value}")
    print(f"\n   📊 Best Sharpe Ratio (Post-Costs): {result['best_sharpe']:.4f}")
    print(f"   ⚡ Average Latency (ms/bar): {result['avg_latency_ms']:.4f}")
    print("=" * 60)
    
    return result


def _generate_synthetic_data(days: int = 180) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(
        end=datetime.now(),
        periods=days,
        freq='D'
    )
    
    # Random walk with drift
    returns = np.random.normal(0.0005, 0.02, days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'high': prices * (1 + np.random.uniform(0, 0.02, days)),
        'low': prices * (1 - np.random.uniform(0, 0.02, days)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
    return data


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_backtest(
    df: pd.DataFrame,
    strategy: Optional[HybridStrategy] = None,
    initial_capital: float = 100000.0,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run backtest on historical data.
    
    Args:
        df: DataFrame with OHLCV data
        strategy: HybridStrategy instance (creates default if None)
        initial_capital: Starting capital
        start_date: Optional start date
        end_date: Optional end date
        **kwargs: Additional parameters
        
    Returns:
        Dict with performance metrics
    """
    if strategy is None:
        strategy = HybridStrategy()
    
    backtester = SimpleBacktester(
        strategy=strategy,
        initial_capital=initial_capital,
        **kwargs
    )
    
    return backtester.run(df, start_date, end_date)


# =============================================================================
# BACKTRADER INTEGRATION (Optional)
# =============================================================================

if BACKTRADER_AVAILABLE:
    
    class HybridBacktraderStrategy(bt.Strategy):
        """Backtrader wrapper for HybridStrategy."""
        
        params = (
            ('momentum_lookback', 20),
            ('bb_std_dev_multiplier', 2.0),
            ('signal_threshold', 0.15),
            ('stop_loss_pct', 0.02),
            ('position_size_pct', 0.10),
        )
        
        def __init__(self):
            self.strategy = HybridStrategy(
                momentum_lookback=self.params.momentum_lookback,
                bb_std_dev_multiplier=self.params.bb_std_dev_multiplier,
                signal_threshold=self.params.signal_threshold
            )
            self.order = None
            self.entry_price = None
        
        def next(self):
            if self.order:
                return
            
            closes = [self.data.close[-i] for i in range(min(len(self), 50))][::-1]
            df = pd.DataFrame({'close': closes})
            
            if len(df) < self.params.momentum_lookback:
                return
            
            signal, _, _ = self.strategy.calculate_signal(df)
            current_price = self.data.close[0]
            
            if self.position and self.entry_price:
                pl_pct = (current_price - self.entry_price) / self.entry_price
                if pl_pct <= -self.params.stop_loss_pct:
                    self.order = self.sell()
                    return
            
            action = self.strategy.get_trade_signal(signal)
            
            if action == 'BUY' and not self.position:
                size = int(
                    (self.broker.getcash() * self.params.position_size_pct) /
                    current_price
                )
                if size > 0:
                    self.order = self.buy(size=size)
                    self.entry_price = current_price
            
            elif action in ('SELL', 'CLOSE') and self.position:
                self.order = self.sell()
                self.entry_price = None
        
        def notify_order(self, order):
            if order.status in [order.Completed]:
                self.order = None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Run walk-forward optimization when executed directly
    run_optimization_walk_forward()
