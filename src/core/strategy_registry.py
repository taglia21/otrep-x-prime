"""
strategy_registry.py
Central registry that maps strategy names to their classes.
Ensures strategies can be dynamically loaded for backtesting or live trading.
"""

from strategies.stoichiometric_strategy import StoichiometricStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy


# Strategy lookup map
STRATEGY_MAP = {
    "StoichiometricStrategy": StoichiometricStrategy,
    "MomentumStrategy": MomentumStrategy,
    "MeanReversionStrategy": MeanReversionStrategy,
}


def get_strategy(name, *args, **kwargs):
    """Return a single strategy instance by name."""
    if name not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {name}")
    return STRATEGY_MAP[name](*args, **kwargs)


def get_strategies(names, *args, **kwargs):
    """Return a list of initialized strategies."""
    return [get_strategy(name, *args, **kwargs) for name in names]
