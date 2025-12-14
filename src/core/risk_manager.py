"""
risk_manager.py
Implements centralized trade risk management for OTREP-X PRIME.
Handles stop-loss, take-profit, volatility targeting, and max exposure limits.
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, max_position=100, stop_loss=0.02, take_profit=0.05, vol_target=0.02):
        """
        Args:
            max_position (int): maximum number of shares per trade
            stop_loss (float): fractional loss threshold to trigger exit
            take_profit (float): fractional profit threshold to trigger exit
            vol_target (float): target daily volatility for position sizing
        """
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.vol_target = vol_target

        # Track open positions: {symbol: {"entry":price, "size":qty}}
        self.positions = {}

    # ---------------- Core Methods ---------------- #

    def size_position(self, symbol, recent_returns):
        """Compute volatility-targeted position size."""
        vol = np.std(recent_returns) if len(recent_returns) > 1 else 0.02
        scale = self.vol_target / max(vol, 1e-6)
        position = int(np.clip(scale * self.max_position, 1, self.max_position))
        log.info(f"[RISK] {symbol}: vol={vol:.4f}, position={position}")
        return position

    def register_entry(self, symbol, price, qty):
        """Record a new position entry."""
        self.positions[symbol] = {"entry": price, "size": qty}
        log.info(f"[RISK] Registered position for {symbol} @ {price} x {qty}")

    def check_exit(self, symbol, current_price):
        """Check stop-loss / take-profit conditions."""
        if symbol not in self.positions:
            return False

        entry = self.positions[symbol]["entry"]
        change = (current_price - entry) / entry

        if change <= -self.stop_loss:
            log.info(f"[RISK] {symbol}: STOP-LOSS triggered ({change*100:.2f}%)")
            del self.positions[symbol]
            return True

        if change >= self.take_profit:
            log.info(f"[RISK] {symbol}: TAKE-PROFIT triggered ({change*100:.2f}%)")
            del self.positions[symbol]
            return True

        return False

    def validate_order(self, price, qty, pnl=None):
        """Legacy pre-trade check used by prototype executors.

        This does not place orders; it only validates basic constraints.
        """
        try:
            if price is None or float(price) <= 0:
                return False
            if abs(int(qty)) > int(self.max_position):
                return False
        except Exception:
            return False
        return True
