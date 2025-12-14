"""Compatibility shim for importing `core.*` from repo root.

This repo uses a `src/` layout but some modules/entrypoints import `core.*`.
Until packaging is introduced, extend the package search path to include `src/core`.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_src_core = Path(__file__).resolve().parent.parent / "src" / "core"
if _src_core.is_dir():
    __path__.append(str(_src_core))  # type: ignore[attr-defined]
