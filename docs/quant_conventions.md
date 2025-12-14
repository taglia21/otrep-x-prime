# Quant Conventions (Canonical)

This document defines **time semantics, portfolio math, cost modeling, and metrics** that the codebase must follow.

## 1) Time & Alignment

### 1.1 Bar timestamps
- All bar timestamps are interpreted as the **bar close time** in an exchange-local timezone unless explicitly stated.
- Internally, timestamps should be normalized to **UTC** for storage and computation.

### 1.2 Decision time vs execution time (no look-ahead)
- A strategy observes information available **up to and including bar $t$ close**.
- Any trade decided at time $t$ is executed at the **next bar** ($t+1$) using the configured fill price.

Default backtest convention for close-only data:
- Signal at $t$ uses prices through **close($t$)**.
- Fills occur at **close($t+1$)**.

This enforces a one-bar delay and prevents look-ahead.

### 1.3 Missing data & alignment
- Price panels are aligned on a shared timestamp index.
- If a symbol price is missing at a fill timestamp, that symbol’s trade is **skipped** for that step.
- Missing-data counts must be recorded as part of run diagnostics.

## 2) Returns & Portfolio Math

### 2.1 Portfolio state
For each timestamp $t$:
- Holdings (shares): $q_{i,t}$
- Cash: $cash_t$
- Mark price: $P_{i,t}$

Portfolio value:
$$
V_t = cash_t + \sum_i q_{i,t} P_{i,t}
$$

### 2.2 Trades and cash updates
A trade at time $t$ changes shares by $\Delta q_{i,t}$ and fills at $F_{i,t}$.

Cash update including costs:
$$
cash_{t^+} = cash_t - \sum_i \Delta q_{i,t} F_{i,t} - cost_t
$$

where $cost_t$ includes commissions and slippage/impact.

### 2.3 Returns
Simple return:
$$r_t = \frac{V_t}{V_{t-1}} - 1$$

Log return:
$$\ell_t = \ln(V_t) - \ln(V_{t-1})$$

Unless explicitly stated, reported strategy performance uses **simple returns** derived from the equity curve.

### 2.4 Reconciliation checks (must hold)
- Equity must reconcile each step: computed $V_t$ equals cash plus marked holdings.
- Trades must reconcile: position delta equals filled shares.

## 3) Cost Model (Configurable)

All costs must be **switchable** and recorded.

### 3.1 Commission
Support at least:
- Fixed + per-share: $a + b\,|shares|$
- (Optional) bps of notional

### 3.2 Slippage / spread
Support at least:
- bps of notional: $slip = s_{bps} \cdot |notional|$
- (Optional) half-spread model

### 3.3 Market impact (optional)
A common parametric form:
$$impact = k \cdot \left(\frac{|notional|}{ADV}\right)^\alpha$$

## 4) Metrics (Must Implement + Unit Test)

Given a return stream $r_t$:
- Volatility (annualized)
- Sharpe (annualized)
- Sortino (annualized)
- Max drawdown
- Calmar ratio
- CAGR
- Hit rate, avg win/loss, profit factor
- Turnover, gross/net exposure, leverage
- Historical VaR and ES at 95% and 99%

Risk-tail conventions:
- Define loss as $L_t = -r_t$.
- VaR at level $\alpha$: $\mathrm{VaR}_\alpha = Q_\alpha(L)$ (a positive loss threshold).
- ES at level $\alpha$: $\mathrm{ES}_\alpha = \mathbb{E}[L \mid L \ge \mathrm{VaR}_\alpha]$.
