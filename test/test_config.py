"""
Test Configuration Loader
=========================
Unit tests for unified configuration loading.

Author: OTREP-X Development Team
Date: December 2025
"""

import pytest
import warnings
import tempfile
import os

import sys
sys.path.insert(0, '.')

from core.config import (
    Config,
    StrategyConfig,
    GraphAlphaConfig,
    RiskConfig,
    MarketFilterConfig,
    load_config,
    get_default_config,
)


class TestConfigLoader:
    """Tests for configuration loading."""
    
    def test_load_config_from_yaml(self):
        """Test loading config from the actual config.yaml."""
        config = load_config('config.yaml')
        
        assert isinstance(config, Config)
        assert len(config.symbols) > 0
        assert config.strategy.signal_threshold > 0
    
    def test_config_strategy_section(self):
        """Test strategy configuration is parsed correctly."""
        config = load_config('config.yaml')
        
        assert isinstance(config.strategy, StrategyConfig)
        assert config.strategy.momentum_lookback == 20
        assert config.strategy.signal_threshold == 0.15
        assert config.strategy.momentum_weight == 0.4
        assert config.strategy.graph_weight == 0.2
    
    def test_config_graph_alpha_section(self):
        """Test graph alpha configuration is parsed correctly."""
        config = load_config('config.yaml')
        
        assert isinstance(config.graph_alpha, GraphAlphaConfig)
        assert config.graph_alpha.adjacency_threshold == 0.5
        assert config.graph_alpha.diffusion_alpha == 1.0
        assert config.graph_alpha.lookback_bars == 50
    
    def test_config_risk_section(self):
        """Test risk configuration is parsed correctly."""
        config = load_config('config.yaml')
        
        assert isinstance(config.risk, RiskConfig)
        assert config.risk.target_risk_per_trade_usd == 125.0
        assert config.risk.max_position_size_usd == 5000.0
        assert config.risk.kill_switch_drawdown_pct == 0.004
    
    def test_config_market_filter_section(self):
        """Test market filter configuration is parsed correctly."""
        config = load_config('config.yaml')
        
        assert isinstance(config.market_filter, MarketFilterConfig)
        assert config.market_filter.spy_correlation_threshold == 0.75
        assert config.market_filter.market_proxy_symbol == 'SPY'
    
    def test_config_backward_compatibility_aliases(self):
        """Test backward compatibility property aliases."""
        config = load_config('config.yaml')
        
        # Check aliases work
        assert config.symbols == config.system.symbols
        assert config.bar_size == config.strategy.timeframe
        assert config.graph_params == config.graph_alpha
        assert config.alpaca_base_url == config.alpaca.base_url
    
    def test_config_graph_alpha_correlation_threshold_alias(self):
        """Test deprecated correlation_threshold property."""
        config = load_config('config.yaml')
        
        # correlation_threshold should alias to adjacency_threshold
        assert config.graph_alpha.correlation_threshold == config.graph_alpha.adjacency_threshold


class TestConfigMigration:
    """Tests for deprecated key migration."""
    
    def test_migrate_correlation_threshold_warning(self):
        """Test deprecation warning for old CORRELATION_THRESHOLD key."""
        # Create a temp config with old key names
        yaml_content = """
SYSTEM:
  SYMBOLS: ['AAPL', 'MSFT']
  CHECK_INTERVAL: 60

GRAPH_ALPHA:
  ENABLED: true
  CORRELATION_THRESHOLD: 0.5
  DIFFUSION_ALPHA: 1.0
  LOOKBACK_BARS: 50

RISK:
  POSITION_SIZE_USD: 500
  TARGET_RISK_PER_TRADE_USD: 125
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                config = load_config(temp_path)
                
                # Should have deprecation warnings
                deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(deprecation_warnings) >= 1
                
                # Config should still work
                assert config.graph_alpha.adjacency_threshold == 0.5
        finally:
            os.unlink(temp_path)


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.yaml')
    
    def test_config_empty_symbols_warning(self):
        """Test validation error for empty symbols."""
        yaml_content = """
SYSTEM:
  SYMBOLS: []
  CHECK_INTERVAL: 60
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            
            assert 'SYMBOLS cannot be empty' in str(exc_info.value)
        finally:
            os.unlink(temp_path)


class TestDefaultConfig:
    """Tests for default configuration."""
    
    def test_get_default_config(self):
        """Test getting default config without YAML file."""
        config = get_default_config()
        
        assert isinstance(config, Config)
        assert len(config.symbols) > 0
        assert config.risk.target_risk_per_trade_usd == 125.0
    
    def test_default_config_has_required_fields(self):
        """Test default config has all required fields."""
        config = get_default_config()
        
        # Check all sections exist
        assert config.system is not None
        assert config.strategy is not None
        assert config.graph_alpha is not None
        assert config.risk is not None
        assert config.alpaca is not None
        assert config.polygon is not None
        assert config.backtest is not None


class TestConfigFromYaml:
    """Tests for Config.from_yaml classmethod."""
    
    def test_from_yaml_classmethod(self):
        """Test Config.from_yaml loads correctly."""
        config = Config.from_yaml('config.yaml')
        
        assert isinstance(config, Config)
        assert len(config.symbols) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
