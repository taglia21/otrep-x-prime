"""
portfolio_manager.py
Consolidates all trades across markets and manages portfolio-level hedging.
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict

log = logging.getLogger(__name__)


class PortfolioManager:
    def __init__(self, hedge_threshold=0.6):
        self.positions = defaultdict(lambda: {"qty": 0, "avg_price": 0.0})
        self.trade_history = []
        self.hedge_threshold = hedge_threshold  # correlation trigger

    # ------------------------------------------------------

    def update_position(self, symbol, qty, price, market):
        pos = self.positions[symbol]
        new_qty = pos["qty"] + qty
        if new_qty != 0:
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + price * qty) / new_qty
        else:
            pos["avg_price"] = 0.0
        pos["qty"] = new_qty
        self.trade_history.append(
            {"symbol": symbol, "market": market, "qty": qty, "price": price}
        )
        log.info(f"[PORTFOLIO] Updated {symbol} → Qty={new_qty}, Avg={pos['avg_price']:.2f}")

    # ------------------------------------------------------

    def consolidate(self):
        df = pd.DataFrame(self.trade_history)
        if df.empty:
            return None
        summary = df.groupby("symbol").agg({"qty": "sum", "price": "mean"}).reset_index()
        log.info(f"[PORTFOLIO] Consolidated holdings:\n{summary}")
        return summary

    # ------------------------------------------------------

    def compute_correlation_matrix(self, price_data: pd.DataFrame):
        """
        Expects DataFrame with columns as symbols and rows as price history.
        """
        returns = price_data.pct_change().dropna()
        corr = returns.corr()
        log.info(f"[PORTFOLIO] Correlation matrix:\n{corr}")
        return corr

    # ------------------------------------------------------

    def apply_correlation_hedge(self, corr: pd.DataFrame):
        """
        Hedge assets with correlation above threshold using inverse positions.
        """
        high_corr_pairs = [
            (a, b)
            for a in corr.columns
            for b in corr.columns
            if a != b and corr.loc[a, b] > self.hedge_threshold
        ]
        if not high_corr_pairs:
            log.info("[HEDGE] No correlations above threshold.")
            return

        for a, b in high_corr_pairs:
            qa = self.positions[a]["qty"]
            qb = self.positions[b]["qty"]
            if qa * qb > 0:  # same direction exposure
                hedge_qty = min(abs(qa), abs(qb))
                self.positions[a]["qty"] -= np.sign(qa) * hedge_qty // 2
                self.positions[b]["qty"] -= np.sign(qb) * hedge_qty // 2
                log.info(f"[HEDGE] Applied cross-hedge between {a} and {b} ({corr.loc[a,b]:.2f})")

    # ------------------------------------------------------

    def portfolio_statistics(self, price_data: pd.DataFrame):
        returns = price_data.pct_change().dropna()
        cov = returns.cov()
        mean_returns = returns.mean()
        weights = np.array([self.positions[s]["qty"] for s in price_data.columns])
        if np.sum(np.abs(weights)) == 0:
            log.info("[PORTFOLIO] No active positions.")
            return
        weights = weights / np.sum(np.abs(weights))
        port_ret = np.dot(mean_returns, weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe = port_ret / port_vol if port_vol > 0 else 0
        log.info(f"[STATS] Portfolio Return={port_ret:.4f}, Vol={port_vol:.4f}, Sharpe={sharpe:.2f}")
