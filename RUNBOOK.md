# RUNBOOK (Codespace)

This repo currently uses a `src/` layout without packaging (packaging planned).

## Safe defaults
- Orders are blocked unless you explicitly opt in.
- `KILL_SWITCH=1` forces the engines to exit non-zero.

## Common commands (from repo root)

Doctor (safe, no secrets printed):
- `PYTHONPATH=src python -m core.doctor`

Doctor broker-state check (read-only):
- `PYTHONPATH=src python -m core.doctor --check-broker-state`

Stage06 live loop (paper-first; still blocked unless enabled):
- `PYTHONPATH=src python -m core.app_stage06_live`

Stage07 async stream (paper-first; still blocked unless enabled):
- `PYTHONPATH=src python -m core.app_stage07_async`

## Enabling paper orders (explicit)
To intentionally allow paper orders:
- `ALLOW_ORDERS=1`
- `NO_ORDERS=0`

## Live trading (explicit multi-ack)
To allow live endpoint order submission you must also set:
- `I_UNDERSTAND_LIVE_TRADING=1`

(And you should also ensure your broker client is configured to use the live endpoint explicitly.)
