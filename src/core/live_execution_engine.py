"""live_execution_engine.py

Production hardening (paper-first):
- Safety: `NO_ORDERS=1` blocks order submission by default.
- Reconciliation: pulls broker account/positions/open orders regularly.
- Order lifecycle: idempotent `client_order_id` + local SQLite journal.
- Market data: uses recent bars (no random sizing inputs).
- Runtime stops: max drawdown, order rejects, consecutive exceptions.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.alpaca_rest import AlpacaRestClient
from core.order_intent_journal import OrderIntentJournal
from core.jsonl_journal import JsonlJournal
from core.market_data import MarketData
from core.order_journal import OrderJournal
from core.order_manager import OrderManager
from core.portfolio_manager import PortfolioManager
from core.reconciler import BrokerReconciler
from core.safety import assert_not_killed
from risk.manager import RiskLimits as CoreRiskLimits
from risk.manager import RiskManager as CoreRiskManager
from core.strategy_registry import get_strategies
from core.ml_optimizer import MLOptimizer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


@dataclass
class RuntimeCounters:
    consecutive_exceptions: int = 0
    order_reject_ts: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.order_reject_ts is None:
            self.order_reject_ts = []


class LiveExecutionEngine:
    def __init__(self, config: dict):
        self.cfg = dict(config)

        run_id = datetime.now(timezone.utc).strftime("live_%Y%m%dT%H%M%SZ")
        self.run_id = run_id

        mode = str(self.cfg.get("mode", "paper")).lower()
        self.paper = mode != "live"

        self.alpaca = AlpacaRestClient(paper=self.paper)
        self.portfolio = PortfolioManager()

        tol = float(self.cfg.get("reconcile_position_mismatch_tolerance", 0.0))
        self.reconciler = BrokerReconciler(
            alpaca=self.alpaca,
            portfolio=self.portfolio,
            position_mismatch_tolerance=tol,
        )

        journal_path = Path(str(self.cfg.get("order_journal_path", "runs/order_journal.sqlite")))
        self.order_journal = OrderJournal(journal_path)

        intent_path = Path(
            str(self.cfg.get("order_intent_journal_path", f"runs/{run_id}/order_intents.jsonl"))
        )
        self.intent_journal = OrderIntentJournal(intent_path)
        self.orders = OrderManager(alpaca=self.alpaca, journal=self.order_journal, intent_journal=self.intent_journal)

        md_cfg = self.cfg.get("market_data", {}) or {}
        self.market_data = MarketData(
            alpaca=self.alpaca,
            timeframe=str(md_cfg.get("timeframe", "1Min")),
            bars_limit=int(md_cfg.get("bars_limit", 200)),
        )

        events_path = Path(str(self.cfg.get("events_path", f"runs/{run_id}/events.jsonl")))
        self.events = JsonlJournal(events_path)

        limits_cfg = self.cfg.get("risk_limits", {}) or {}
        self.limits = CoreRiskLimits(
            max_gross_leverage=float(limits_cfg.get("max_gross_leverage", 1.0)),
            max_position_pct=float(limits_cfg.get("max_position_pct", 0.10)),
            max_notional_per_symbol=float(limits_cfg.get("max_notional_per_symbol", 50_000.0)),
            max_turnover_per_step=float(limits_cfg.get("max_turnover_per_step", 0.25)),
        )
        self.risk = CoreRiskManager(self.limits)

        self.max_daily_drawdown_pct = float(self.cfg.get("max_daily_drawdown_pct", 0.10))
        self.max_order_rejects_per_hour = int(self.cfg.get("max_order_rejects_per_hour", 5))
        self.max_consecutive_exceptions = int(self.cfg.get("max_consecutive_exceptions", 5))

        self._day = None
        self._day_peak_equity: float | None = None

        self.counters = RuntimeCounters()

        self.strategies = get_strategies(self.cfg["strategies"])
        self.optimizer = MLOptimizer(self.cfg["strategies"], learning_rate=0.2)

        self.symbols = (
            self.cfg["markets"]["US"]["symbols"]
            + self.cfg["markets"]["EU"]["symbols"]
            + self.cfg["markets"]["ASIA"]["symbols"]
        )

        sizing_cfg = self.cfg.get("sizing", {}) or {}
        self.target_notional_pct = float(sizing_cfg.get("target_notional_pct", 0.02))
        self.target_vol_per_bar = float(sizing_cfg.get("target_vol_per_bar", 0.01))
        self.min_vol = float(sizing_cfg.get("min_vol", 1e-4))
        self.min_scale = float(sizing_cfg.get("min_scale", 0.25))
        self.max_scale = float(sizing_cfg.get("max_scale", 2.0))

    def _size_from_stats(self, *, equity: float, price: float, realized_vol: float) -> int | None:
        if equity <= 0 or price <= 0:
            return None

        if realized_vol != realized_vol:  # NaN
            return None

        vol = float(realized_vol)
        if vol < self.min_vol:
            vol = self.min_vol

        scale = self.target_vol_per_bar / vol
        scale = max(self.min_scale, min(self.max_scale, scale))

        base_notional = float(equity) * self.target_notional_pct
        cap_notional = min(
            float(self.limits.max_notional_per_symbol),
            float(equity) * float(self.limits.max_position_pct),
        )
        target_notional = min(base_notional * scale, cap_notional)

        qty = int(target_notional // float(price))
        if qty <= 0:
            return None
        return qty

    def _update_daily_drawdown(self, *, equity: float) -> None:
        today = datetime.now(timezone.utc).date()
        if self._day != today:
            self._day = today
            self._day_peak_equity = float(equity)

        if self._day_peak_equity is None:
            self._day_peak_equity = float(equity)

        self._day_peak_equity = max(float(self._day_peak_equity), float(equity))
        peak = float(self._day_peak_equity)
        if peak <= 0:
            return
        dd = (peak - float(equity)) / peak
        if dd >= self.max_daily_drawdown_pct:
            log.error(f"[RISK] max_daily_drawdown_pct exceeded dd={dd:.4f} >= {self.max_daily_drawdown_pct}")
            raise SystemExit(2)

    def _update_order_rejects(self) -> None:
        now = time.time()
        self.counters.order_reject_ts = [t for t in self.counters.order_reject_ts if now - t <= 3600]
        if len(self.counters.order_reject_ts) > self.max_order_rejects_per_hour:
            log.error("[RISK] max_order_rejects_per_hour exceeded")
            raise SystemExit(2)

    def run(self, interval: int = 60) -> None:
        log.info("[ENGINE] Starting real-time trading loop …")

        # Startup reconciliation is read-only.
        state = self.reconciler.reconcile()
        equity = float(state.account.equity)
        self._update_daily_drawdown(equity=equity)

        self.events.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "startup_reconcile",
                "broker_mode": "paper" if self.paper else "live",
                "equity": equity,
                "cash": float(state.account.cash),
                "positions": len(state.positions),
                "open_orders": len(state.open_orders),
            }
        )

        # Reconcile broker open orders into local journal for idempotency.
        try:
            self.orders.reconcile_open_orders()
            self.events.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "startup_reconcile_open_orders",
                }
            )
        except Exception as e:
            self.events.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "startup_reconcile_open_orders_failed",
                    "reason": type(e).__name__,
                }
            )

        while True:
            try:
                assert_not_killed()

                state = self.reconciler.reconcile()
                equity = float(state.account.equity)
                self._update_daily_drawdown(equity=equity)
                self._update_order_rejects()

                for symbol in self.symbols:
                    assert_not_killed()

                    stats = self.market_data.recent_returns_and_vol(symbol=symbol)
                    if not stats.returns or stats.last_price is None:
                        continue

                    price = float(stats.last_price)
                    size = self._size_from_stats(equity=equity, price=price, realized_vol=float(stats.realized_vol))
                    if size is None:
                        continue

                    # Existing strategies expect a single scalar "price".
                    signals = {s.__class__.__name__: s.generate_signal(price) for s in self.strategies}
                    action = self.optimizer.choose_action(signals)

                    if action > 0:
                        decision = self.risk.check_order(equity=equity, symbol=symbol, qty=size, price=price)
                        if not decision.allowed:
                            self.events.append(
                                {
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                    "event": "risk_block",
                                    "symbol": symbol,
                                    "side": "buy",
                                    "reasons": decision.reasons,
                                }
                            )
                            continue

                        od = self.orders.submit_order(strategy="live", symbol=symbol, qty=size, side="buy")
                        self.events.append(
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "event": "order_decision",
                                "symbol": symbol,
                                "side": "buy",
                                "qty": size,
                                "client_order_id": od.client_order_id,
                                "submitted": od.submitted,
                                "reason": od.reason,
                            }
                        )
                        if od.reason.startswith("submit_failed"):
                            self.counters.order_reject_ts.append(time.time())

                    elif action < 0:
                        decision = self.risk.check_order(equity=equity, symbol=symbol, qty=size, price=price)
                        if not decision.allowed:
                            self.events.append(
                                {
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                    "event": "risk_block",
                                    "symbol": symbol,
                                    "side": "sell",
                                    "reasons": decision.reasons,
                                }
                            )
                            continue

                        od = self.orders.submit_order(strategy="live", symbol=symbol, qty=size, side="sell")
                        self.events.append(
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "event": "order_decision",
                                "symbol": symbol,
                                "side": "sell",
                                "qty": size,
                                "client_order_id": od.client_order_id,
                                "submitted": od.submitted,
                                "reason": od.reason,
                            }
                        )
                        if od.reason.startswith("submit_failed"):
                            self.counters.order_reject_ts.append(time.time())

                self.optimizer.adapt_weights()
                self.counters.consecutive_exceptions = 0
                time.sleep(interval)
            except KeyboardInterrupt:
                log.warning("[ENGINE] Stopped manually.")
                break
            except SystemExit as e:
                log.error(f"[ENGINE] exit code={e.code}")
                raise
            except Exception as e:
                self.counters.consecutive_exceptions += 1
                if self.counters.consecutive_exceptions >= self.max_consecutive_exceptions:
                    log.error("[ENGINE] max_consecutive_exceptions exceeded")
                    raise SystemExit(2)
                log.error(f"[ENGINE] {type(e).__name__}")
                time.sleep(interval)
