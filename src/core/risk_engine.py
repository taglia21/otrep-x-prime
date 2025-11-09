import numpy as np

class RiskEngine:
    def __init__(self, max_drawdown=0.15, base_position=1.0):
        self.max_drawdown = max_drawdown
        self.base_position = base_position
        self.equity_curve = [1.0]
        self.position_size = base_position

    def update_equity(self, pnl_pct: float):
        new_equity = self.equity_curve[-1] * (1 + pnl_pct)
        self.equity_curve.append(new_equity)
        drawdown = 1 - new_equity / max(self.equity_curve)
        if drawdown > self.max_drawdown:
            self.position_size *= 0.5  # halve exposure
        return self.position_size

    def get_metrics(self):
        returns = np.diff(self.equity_curve)
        mean = np.mean(returns)
        std = np.std(returns)
        sharpe = mean / std * np.sqrt(252) if std > 0 else 0
        max_dd = 1 - min(self.equity_curve) / max(self.equity_curve)
        return {"Sharpe": sharpe, "MaxDrawdown": max_dd}
