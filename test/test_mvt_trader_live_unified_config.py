"""Tests for mvt_trader_live unified-config wiring."""

import sys

import pytest

sys.path.insert(0, '.')


def test_config_graph_alpha_threshold_is_0_5():
    from core.config import load_config

    cfg = load_config('config.yaml')
    assert cfg.graph_alpha.adjacency_threshold == 0.5


def test_mvt_trader_graph_params_uses_adjacency_threshold():
    import otrep_core
    from core.config import load_config
    import mvt_trader_live

    cfg = load_config('config.yaml')
    trader = mvt_trader_live.MVTTrader(cfg, logger=lambda *_: None, skip_account_sync=True)
    params = trader.cpp_graph_params

    assert isinstance(params, otrep_core.GraphParams)
    assert params.correlation_threshold == 0.5
