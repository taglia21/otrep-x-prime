from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Iterable

import yaml


def _git(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(["git", *cmd], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _env_status(name: str) -> str:
    v = os.getenv(name)
    if v is None:
        return f"{name}=UNSET"
    return f"{name}=SET len={len(v)}"


def _print_env(names: Iterable[str]) -> None:
    for n in names:
        print(_env_status(n))


def _check_config(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            print(f"config_load=FAIL path={path} reason=not_a_mapping")
            return 2
        print(f"config_load=OK path={path}")
        return 0
    except FileNotFoundError:
        print(f"config_load=FAIL path={path} reason=missing")
        return 2
    except Exception as e:
        print(f"config_load=FAIL path={path} reason={type(e).__name__}")
        return 2


def _check_alpaca_quote(symbol: str) -> int:
    # Read-only quote request. Never place orders.
    try:
        import requests
    except Exception:
        print("alpaca_check=FAIL reason=requests_missing")
        return 2

    key = os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        print("alpaca_check=SKIP reason=missing_env")
        return 2

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            print(f"alpaca_check=FAIL status={r.status_code}")
            return 2
        q = r.json().get("quote", {})
        price = q.get("ap") or q.get("bp")
        print(f"alpaca_check=OK symbol={symbol} quote_price={price}")
        return 0
    except Exception as e:
        print(f"alpaca_check=FAIL reason={type(e).__name__}")
        return 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="otrep-doctor", add_help=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--check-alpaca", action="store_true")
    p.add_argument("--symbol", default="AAPL")
    args = p.parse_args(argv)

    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    sha = _git(["rev-parse", "HEAD"])

    print(f"git_branch={branch}")
    print(f"git_sha={sha}")

    rc = 0
    rc = max(rc, _check_config(args.config))

    # Required env names for live/paper connectivity
    _print_env(["ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"])

    if args.check_alpaca:
        rc = max(rc, _check_alpaca_quote(args.symbol))
    else:
        print("alpaca_check=SKIP (pass --check-alpaca)")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
