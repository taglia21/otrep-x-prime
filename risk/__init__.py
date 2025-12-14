"""Compatibility shim for importing `risk.*` from repo root.

This repo uses a `src/` layout but some entrypoints import `risk.*`.
Until packaging is introduced (PR3), extend the package search path to include `src/risk`.
"""

from __future__ import annotations

from pkgutil import extend_path
from pathlib import Path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_src_risk = Path(__file__).resolve().parent.parent / "src" / "risk"
if _src_risk.is_dir():
    __path__.append(str(_src_risk))  # type: ignore[attr-defined]
