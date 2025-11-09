import random
from src.core_logger import logger
from src.core.risk_manager import RiskManager

class TradeExecutor:
    def __init__(self, mode="paper", risk_cfg=None):
        self.mode = mode
        self.risk = RiskManager(**(risk_cfg or {}))
        logger.info(f"TradeExecutor initialized in {mode.upper()} mode")

    def execute(self, signal, symbol="AAPL", price=0):
        qty = 10 if signal > 0 else -10
        pnl = round(random.uniform(-0.01, 0.03), 4)  # mock P&L for now
        if not self.risk.validate_order(price, qty, pnl):
            logger.warning(f"Order for {symbol} skipped due to risk filters")
            return None
        side = "BUY" if signal > 0 else "SELL"
        logger.info(f"Executed {side} order for {symbol} ({qty}) @ {price}")
        return {"symbol": symbol, "side": side, "qty": qty, "price": price, "pnl": pnl}
