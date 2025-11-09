"""
ml_optimizer.py
Adaptive meta-learning layer for OTREP-X PRIME.
Uses reinforcement learning–style feedback to re-weight strategies dynamically.
"""

import numpy as np
import logging
from collections import deque

log = logging.getLogger(__name__)


class MLOptimizer:
    def __init__(self, strategy_names, learning_rate=0.1, window=20):
        self.strategy_names = strategy_names
        self.learning_rate = learning_rate
        self.window = window
        self.performance = {s: deque(maxlen=window) for s in strategy_names}
        self.weights = {s: 1.0 / len(strategy_names) for s in strategy_names}

    # -----------------------------------------------------------

    def update_performance(self, strategy_name, reward):
        """Record reward (+ for profit, − for loss)."""
        self.performance[strategy_name].append(reward)

    def _normalize(self):
        total = sum(self.weights.values())
        if total > 0:
            for s in self.weights:
                self.weights[s] /= total

    def adapt_weights(self):
        """Adjust weights by recent performance (reinforcement update)."""
        for s, history in self.performance.items():
            if len(history) == 0:
                continue
            avg_reward = np.mean(history)
            self.weights[s] += self.learning_rate * avg_reward
        self._normalize()
        log.info(f"[ML] Updated strategy weights: "
                 + ", ".join(f"{k}={v:.2f}" for k, v in self.weights.items()))

    def choose_action(self, signals):
        """
        Combine strategy signals (−1, 0, +1) into one meta-signal.
        """
        total = 0.0
        for s, signal in signals.items():
            total += self.weights[s] * signal
        return np.sign(total)
