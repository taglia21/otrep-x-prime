import os
import logging
from core.strategy_registry import StrategyRegistry
from core.backtester import Backtester

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Initializing OTREP-X PRIME Backtest Module…")

    strategy_name = os.getenv("STRATEGY_NAME", "StoichiometricStrategy")
    data_path = os.getenv("BACKTEST_DATA", "data/aapl_mock.csv")

    registry = StrategyRegistry(logger)
    strategy_class = registry.load(strategy_name)

    if strategy_class:
        backtester = Backtester(strategy_class, data_path, logger)
        backtester.run()
    else:
        logger.info("Strategy loading failed — aborting backtest.")
