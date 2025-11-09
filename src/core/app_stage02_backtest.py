"""
Phase V – Portfolio Consolidation + Cross-Asset Correlation Hedging
"""

import yaml, logging, pandas as pd
from datetime import datetime
from core.strategy_registry import get_strategies
from core.risk_manager import RiskManager
from core.ml_optimizer import MLOptimizer
from core.multimarket_manager import MultiMarketManager, MarketFeed
from core.portfolio_manager import PortfolioManager

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TradeExecutor:
    def __init__(self, portfolio: PortfolioManager):
        self.portfolio = portfolio

    def execute(self, action, symbol, price, qty, market):
        sign = 1 if action.lower() == "buy" else -1
        self.portfolio.update_position(symbol, sign * qty, price, market)
        log.info(f"{market}: {action.upper()} {symbol} @ {price:.2f} x{qty}")


def generate_market_data():
    base = pd.date_range("2025-10-01", periods=6)
    us = pd.DataFrame({"datetime": base, "symbol": "AAPL", "close": [152.3,147.8,143.2,156.0,201.4,210.6]})
    eu = pd.DataFrame({"datetime": base, "symbol": "BMW.DE", "close": [94.1,96.3,91.5,97.4,101.2,99.8]})
    asia = pd.DataFrame({"datetime": base, "symbol": "7203.T", "close": [2320,2410,2295,2430,2525,2488]})
    return {"US": MarketFeed("US", us, -5), "EU": MarketFeed("EU", eu, 1), "ASIA": MarketFeed("ASIA", asia, 9)}


def main():
    cfg = load_config()
    log.info("Loaded configuration from config.yaml")

    portfolio = PortfolioManager(hedge_threshold=0.7)
    risk = RiskManager(**cfg["risk_management"])
    strategies = get_strategies(cfg["strategies"])
    optimizer = MLOptimizer(cfg["strategies"], learning_rate=0.2)
    feeds = generate_market_data()
    manager = MultiMarketManager(feeds)
    executor = TradeExecutor(portfolio)

    price_cache = {"AAPL": [], "BMW.DE": [], "7203.T": []}

    while True:
        merged = manager.step_all()
        if merged is None:
            break
        for _, row in merged.iterrows():
            market, price, symbol = row["market"], row["close"], row["symbol"]
            recent_returns = [0.01, -0.02, 0.015]
            size = risk.size_position(symbol, recent_returns)

            signals = {s.__class__.__name__: s.generate_signal(price) for s in strategies}
            action = optimizer.choose_action(signals)
            if action > 0:
                executor.execute("buy", symbol, price, size, market)
                risk.register_entry(symbol, price, size)
            elif action < 0 or risk.check_exit(symbol, price):
                executor.execute("sell", symbol, price, size, market)

            price_cache[symbol].append(price)
        optimizer.adapt_weights()

    df_prices = pd.DataFrame(price_cache)
    corr = portfolio.compute_correlation_matrix(df_prices)
    portfolio.apply_correlation_hedge(corr)
    portfolio.portfolio_statistics(df_prices)
    portfolio.consolidate()

    log.info("[SUMMARY] Phase V complete – portfolio consolidated and hedged.")


if __name__ == "__main__":
    main()
