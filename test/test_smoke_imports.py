"""
Smoke Test - Import Verification
=================================
Verify all modules import correctly without errors.

Author: OTREP-X Development Team
Date: December 2025
"""

import pytest
import sys
sys.path.insert(0, '.')


class TestSmokeImports:
    """Smoke tests for module imports."""
    
    def test_import_core_config(self):
        """Test core config module imports."""
        from core.config import Config, load_config, get_default_config
        assert Config is not None
        assert load_config is not None
    
    def test_import_risk_module(self):
        """Test risk module imports."""
        from risk import (
            KillSwitch,
            KillSwitchConfig,
            PositionSizer,
            PositionSizerConfig,
            compute_atr,
            compute_position_size,
        )
        assert KillSwitch is not None
        assert PositionSizer is not None
    
    def test_import_services_module(self):
        """Test services module imports."""
        from services import (
            MarketDataService,
            MarketDataConfig,
            DataStaleError,
            DataFetchError,
        )
        assert MarketDataService is not None
        assert DataStaleError is not None
    
    def test_import_utils_module(self):
        """Test utils module imports."""
        from utils.logger import (
            AuditEventType,
            AuditEvent,
            setup_logging,
        )
        assert AuditEventType is not None
        assert AuditEvent is not None
    
    def test_import_models_module(self):
        """Test models module imports."""
        from models.tca_model import TCAModel, TCAConfig
        assert TCAModel is not None
        assert TCAConfig is not None
    
    def test_import_api_module(self):
        """Test API module imports."""
        from api.polygon_client import PolygonClient
        from api.alpaca_client import AlpacaClient
        assert PolygonClient is not None
        assert AlpacaClient is not None
    
    def test_import_risk_manager(self):
        """Test risk_manager imports."""
        from risk_manager import RiskManager, create_risk_manager_from_config
        assert RiskManager is not None
        assert create_risk_manager_from_config is not None
    
    def test_import_cpp_extension(self):
        """Test C++ extension imports."""
        import otrep_core
        assert hasattr(otrep_core, 'HybridStrategy')
        assert hasattr(otrep_core, 'MarketGraph')
        assert hasattr(otrep_core, 'get_version')
    
    def test_cpp_extension_version(self):
        """Test C++ extension has valid version."""
        import otrep_core
        version = otrep_core.get_version()
        assert version is not None
        assert len(version) > 0


class TestSmokeConfig:
    """Smoke tests for configuration."""
    
    def test_load_actual_config(self):
        """Test loading the actual config.yaml."""
        from core.config import load_config
        
        config = load_config('config.yaml')
        
        assert config is not None
        assert len(config.symbols) > 0
        assert config.graph_alpha.adjacency_threshold == 0.5
    
    def test_config_has_all_sections(self):
        """Test config has all required sections."""
        from core.config import load_config
        
        config = load_config('config.yaml')
        
        assert config.system is not None
        assert config.strategy is not None
        assert config.graph_alpha is not None
        assert config.risk is not None
        assert config.market_filter is not None


class TestSmokeRiskComponents:
    """Smoke tests for risk components."""
    
    def test_kill_switch_instantiation(self):
        """Test kill switch can be instantiated."""
        from risk import KillSwitch, KillSwitchConfig
        
        config = KillSwitchConfig(daily_drawdown_limit_pct=0.004)
        ks = KillSwitch(config=config)
        
        assert ks.is_trading_allowed() is True
    
    def test_position_sizer_instantiation(self):
        """Test position sizer can be instantiated."""
        from risk import PositionSizer, PositionSizerConfig
        
        config = PositionSizerConfig(target_risk_per_trade_usd=125.0)
        sizer = PositionSizer(config=config)
        
        result = sizer.calculate_size(entry_price=100.0, atr=2.0)
        assert result.shares > 0
    
    def test_risk_manager_instantiation(self):
        """Test risk manager can be instantiated."""
        from risk_manager import RiskManager
        
        rm = RiskManager(
            stop_loss_pct=0.02,
            max_positions=3,
            target_risk_per_trade_usd=125.0
        )
        
        assert rm.max_positions == 3


class TestSmokeCppEngine:
    """Smoke tests for C++ engine."""
    
    def test_hybrid_strategy(self):
        """Test C++ HybridStrategy works."""
        import otrep_core
        
        params = otrep_core.StrategyParams()
        params.momentum_lookback = 20
        params.signal_threshold = 0.15
        
        strategy = otrep_core.HybridStrategy(params)
        
        # Feed a bar
        bar = otrep_core.Bar(100.0, 102.0, 99.0, 101.0, 1000)
        strategy.on_bar(bar)
        
        # Should not crash
        assert True
    
    def test_market_graph(self):
        """Test C++ MarketGraph works."""
        import otrep_core
        import numpy as np
        
        graph = otrep_core.MarketGraph()
        params = otrep_core.GraphParams()
        params.correlation_threshold = 0.5
        graph.set_params(params)
        
        # Create dummy returns matrix
        returns = np.random.randn(50, 5).astype(np.float64)
        returns = np.ascontiguousarray(returns)
        
        result = graph.calculate_signals(returns)
        
        assert result is not None
        assert len(result.signals) == 5


class TestSmokeDryRun:
    """Smoke test for minimal dry run."""
    
    def test_minimal_trading_cycle(self):
        """Test a minimal trading cycle without real API calls."""
        import otrep_core
        import numpy as np
        from core.config import load_config
        from risk import KillSwitch, PositionSizer
        from risk_manager import RiskManager
        
        # Load config
        config = load_config('config.yaml')
        
        # Initialize components
        ks = KillSwitch()
        ks.set_starting_equity(100000.0)
        
        rm = RiskManager(
            target_risk_per_trade_usd=config.risk.target_risk_per_trade_usd
        )
        rm.set_starting_equity(100000.0)
        
        sizer = PositionSizer()
        
        # Initialize C++ strategy
        cpp_params = otrep_core.StrategyParams()
        cpp_params.momentum_lookback = config.strategy.momentum_lookback
        cpp_params.signal_threshold = config.strategy.signal_threshold
        strategy = otrep_core.HybridStrategy(cpp_params)
        
        # Initialize graph
        graph = otrep_core.MarketGraph()
        graph_params = otrep_core.GraphParams()
        graph_params.correlation_threshold = config.graph_alpha.adjacency_threshold
        graph.set_params(graph_params)
        
        # Simulate a cycle
        assert ks.is_trading_allowed() is True
        
        # Feed some bars
        for i in range(25):
            bar = otrep_core.Bar(
                100.0 + i * 0.1,
                102.0 + i * 0.1,
                99.0 + i * 0.1,
                101.0 + i * 0.1,
                1000
            )
            strategy.on_bar(bar)
        
        # Get signal
        signal_result = strategy.calculate_signal(0.0)
        
        # Position sizing
        size_result = sizer.calculate_size(entry_price=100.0, atr=2.0)
        
        # Pre-trade check
        check_result = rm.pre_trade_check(
            symbol='AAPL',
            side='BUY',
            entry_price=100.0,
            shares=size_result.shares,
            current_positions=0,
            current_equity=100000.0
        )
        
        # Cycle complete
        assert check_result['allowed'] is True
        assert size_result.shares > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
